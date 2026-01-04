from __future__ import print_function

import json
import os
import random
from types import SimpleNamespace
import numpy as np
from pprint import pprint

import torch
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter

from dataloader import get_dataloader
from cli_parser import parse_args
from metrics import eval_and_log_metrics, eval_and_log_validation_metrics, log_hparams, log_generator_distribution
from my_utils import *
from approximate_gradients import estimate_gradient_objective

from teacher import TeacherModel
from ensemble import Ensemble
from replay import init_replay_memory
from query_selectors.query_selector import select as query_select
from training.generator_trainer import disguide_generator_step
import config


def _build_informativeness_cfg(args):
    return SimpleNamespace(
        use_mc_uncertainty=getattr(args, "use_mc_uncertainty", False),
        mc_threshold=getattr(args, "mc_threshold", None),
        mc_samples=getattr(args, "mc_samples", None),
        use_boundary_sampling=getattr(args, "use_boundary_sampling", False),
        boundary_sample_ratio=getattr(args, "boundary_sample_ratio", None),
        combine_strategy=getattr(args, "informativeness_combine_strategy", "weighted_sum"),
        weighted_sum_cfg=SimpleNamespace(
            w_epistemic=getattr(args, "informativeness_weight_epistemic", None),
            w_boundary=getattr(args, "informativeness_weight_boundary", None),
        ),
        gate_then_rank_cfg=SimpleNamespace(
            epistemic_gate=getattr(args, "informativeness_gate_quantile", None),
            boundary_topk_ratio=getattr(args, "informativeness_boundary_topk", None),
        ),
        differentiable_version=getattr(args, "informativeness_differentiable", False),
        percentiles=getattr(args, "selector_percentiles", [0.5, 0.9]),
    )


def _build_selector_cfg(args):
    return SimpleNamespace(
        use_mc_uncertainty=getattr(args, "use_mc_uncertainty", False),
        mc_threshold=getattr(args, "mc_threshold", None),
        mc_samples=getattr(args, "mc_samples", None),
        use_boundary_sampling=getattr(args, "use_boundary_sampling", False),
        boundary_sample_ratio=getattr(args, "boundary_sample_ratio", None),
        percentiles=getattr(args, "selector_percentiles", [0.5, 0.9]),
    )


def _build_gen_cfg(args):
    return SimpleNamespace(
        use_informativeness_prior=getattr(args, "use_informativeness_prior", False),
        informativeness_weight=getattr(args, "informativeness_weight", 0.0),
        informativeness_schedule=getattr(args, "informativeness_schedule", "none"),
        informativeness_warmup=getattr(args, "informativeness_warmup", 1),
        objective=getattr(args, "informativeness_objective", "mean"),
        topk_ratio=getattr(args, "informativeness_topk_ratio", None),
    )


def adaptive_sampling(fake, student_ensemble, args):
    """Posterior selector wrapper (teacher-free)."""
    selector_cfg = getattr(args, "selector_cfg", args)
    selected, meta = query_select(fake, student_ensemble, selector_cfg)
    indices = meta.get("selected_indices", torch.arange(selected.size(0), device=fake.device))
    if selected.size(0) == 0:
        return fake, torch.arange(fake.size(0), device=fake.device), meta
    return selected, indices, meta


def train_generator(args, generator, student_ensemble, teacher, device, optimizer):
    """Train generator model. Methodology is based on cli input args, especially the experiment-type parameter."""
    assert not teacher.training
    generator.train()
    student_ensemble.eval()

    g_loss_sum = 0
    for i in range(args.g_iter):
        optimizer.zero_grad()
        z = torch.randn((args.batch_size, args.nz)).to(device)
        if args.experiment_type == 'dfme':
            g_loss = dfme_gen_loss(args, z, generator, student_ensemble, teacher, device)
        elif args.experiment_type == 'disguide':
            if not hasattr(args, "gen_update_count"):
                args.gen_update_count = 0
            g_loss, aux = disguide_generator_step(args, z, generator, student_ensemble, args.gen_update_count)
            args.gen_update_count += 1
            if hasattr(config, "tboard_writer") and config.tboard_writer is not None and isinstance(aux, dict):
                if "prior_weight" in aux:
                    config.tboard_writer.add_scalar('Param/informativeness_prior_weight', aux["prior_weight"], args.current_query_count)
        optimizer.step()
        g_loss_sum += g_loss
    return g_loss_sum / args.g_iter


def dfme_gen_loss(args, z, generator, student_ensemble, teacher, device):
    """Compute the generator loss for DFME method. Uses forward differences method. Update weights based on loss.
    See Also: https://github.com/cake-lab/datafree-model-extraction"""
    fake = generator(z, pre_x=args.approx_grad)  # pre_x returns the output of G before applying the activation

    # Estimate gradient for black box teacher
    approx_grad_wrt_x, loss_G = estimate_gradient_objective(args, teacher, student_ensemble, fake,
                                                            epsilon=args.grad_epsilon, m=args.grad_m,
                                                            device=device, pre_x=True)

    fake.backward(approx_grad_wrt_x)
    return loss_G.item()

def supervised_student_training(student_ensemble, fake, t_logit, optimizer, args):
    """Calculate loss and update weights for students in a supervised fashion"""
    student_iter_preds = []
    student_iter_loss = 0
    for i in range(student_ensemble.size()):
        s_logit = student_ensemble(fake, idx=i)
        with torch.no_grad():
            student_iter_preds.append(F.softmax(s_logit, dim=-1).detach())
        loss_s = student_loss(s_logit, t_logit, args)  # Helper function which handles soft- and hard-label settings
        loss_s.backward()
        student_iter_loss += loss_s.item()
    optimizer.step()
    return torch.stack(student_iter_preds, dim=1), student_iter_loss


def train_student_ensemble(args, generator, student_ensemble, teacher, device, optimizer, replay_memory):
    """Train student ensemble with a fixed generator"""
    assert not teacher.training
    generator.eval()
    student_ensemble.train()

    s_loss_sum = 0
    student_preds = []
    teacher_preds = []

    # 记录查询前的计数
    query_count_before = args.current_query_count

    for d_iter in range(args.d_iter):
        optimizer.zero_grad()

        # 生成更多样本用于筛选
        oversample = 2 if (args.use_boundary_sampling or args.use_mc_uncertainty) else 1
        z = torch.randn((args.batch_size * oversample, args.nz)).to(device)
        fake_all = generator(z).detach()

        # 自适应采样
        fake, _, selector_meta = adaptive_sampling(fake_all, student_ensemble, args)
        args.last_selector_meta = selector_meta

        # 确保至少有一半batch_size
        if fake.size(0) < args.batch_size // 2:
            remaining = args.batch_size // 2 - fake.size(0)
            fake = torch.cat([fake, fake_all[:remaining]], dim=0)

        t_logit = get_teacher_prediction(args, teacher, fake)
        replay_memory.update(fake.cpu(), t_logit.cpu())
        student_iter_preds, student_iter_loss = supervised_student_training(student_ensemble, fake, t_logit,
                                                                            optimizer, args)

        teacher_preds.append(F.softmax(t_logit, dim=-1).detach())
        student_preds.append(student_iter_preds)
        s_loss_sum += student_iter_loss

    # 检查replay_memory是否有足够样本
    if hasattr(replay_memory, 'size'):
        replay_available = replay_memory.size >= args.batch_size
    else:
        replay_available = len(replay_memory) >= args.batch_size

    if replay_available:
        for _ in range(args.rep_iter):
            optimizer.zero_grad()
            fake, t_logit = replay_memory.sample()
            fake = fake.to(device)
            t_logit = t_logit.to(device)
            student_iter_preds, student_iter_loss = supervised_student_training(student_ensemble, fake, t_logit,
                                                                                optimizer, args)
            teacher_preds.append(F.softmax(t_logit, dim=-1).detach())
            student_preds.append(student_iter_preds)
            s_loss_sum += student_iter_loss

    teacher_preds = torch.cat(teacher_preds, dim=0)
    student_preds = torch.cat(student_preds, dim=0)

    # 计算实际查询增量
    actual_queries = args.current_query_count - query_count_before

    return student_preds, teacher_preds, s_loss_sum / (args.d_iter * student_ensemble.size()), actual_queries


def log_training_epoch(args, g_loss, s_loss, i, epoch):
    """Somewhat dated function for logging training data to command line and logfile"""
    if i % args.log_interval != 0:
        return
    print_and_log(f'Train Epoch: {epoch} [{i}/{args.epoch_itrs} ({100*float(i)/float(args.epoch_itrs):.0f}%)] '
                  f'Generator Loss:{g_loss} Student Loss:{s_loss}')


def train_epoch_ensemble(args, generator, student_ensemble, teacher, device,
                         optimizer_student, optimizer_generator, epoch, replay_memory):
    """Runs alternating generator and student training iterations."""
    student_preds = None
    teacher_preds = None
    for i in range(args.epoch_itrs):
        g_loss = train_generator(args, generator, student_ensemble, teacher, device, optimizer_generator)
        student_preds, teacher_preds, s_loss, actual_queries = train_student_ensemble(args, generator, student_ensemble,
                                                                                      teacher,
                                                                                      device, optimizer_student,
                                                                                      replay_memory)

        # 使用实际查询数更新预算
        args.query_budget -= actual_queries
        assert (args.query_budget + args.current_query_count == args.total_query_budget), \
            f"{args.query_budget} + {args.current_query_count}"

        log_training_epoch(args, g_loss, s_loss, i, epoch)
        if args.query_budget < args.cost_per_iteration:
            break
    return student_preds, teacher_preds


def log_test_metrics(model, test_loader, teacher_test_preds, device, args):
    if not isinstance(model, Ensemble):
        print_and_log(f"log_test_metrics currently only supports Ensemble. Detected class:{model.__class__}")
        raise NotImplementedError

    preds, labels = get_model_preds_and_true_labels(model, test_loader, device)
    stats = eval_and_log_metrics(config.tboard_writer, preds, labels, args)
    print_and_log(
        'Accuracies=> Soft Vote:{:.4f}%, Hard Vote:{:.4f}%, Es Median/Min/Max:{:.4f}%/{:.4f}%/{:.4f}%\n'.format(
            100 * stats["Soft Vote"]["Accuracy"],
            100 * stats["Hard Vote"]["Accuracy"],
            100 * stats["Ensemble"]["Accuracy"]["Median"],
            100 * stats["Ensemble"]["Accuracy"]["Min"],
            100 * stats["Ensemble"]["Accuracy"]["Max"]))
    eval_and_log_validation_metrics(config.tboard_writer, preds, teacher_test_preds, args, tag="Fidelity")
    return stats["Soft Vote"]["Accuracy"]


def get_model_accuracy_and_loss(args, model, loader, device="cuda"):
    """Get model accuracy and loss. Simple function intended for simply CLI printing. Prefer using metrics.py"""
    model.eval()
    correct, loss = 0, 0
    with torch.no_grad():
        for i, (data, target) in enumerate(loader):
            data, target = data.to(device), target.to(device)
            output = model(data)
            pred = output.argmax(dim=1, keepdim=True)  # get the index of the max log-probability
            correct += pred.eq(target.view_as(pred)).sum().item()
            loss += F.cross_entropy(output, target, reduction='sum').item()  # sum up batch
    loss /= len(loader.dataset)
    accuracy = correct / len(loader.dataset)
    return accuracy, loss


def _json_default(obj):
    try:
        import torch
    except Exception:
        torch = None

    if torch is not None and isinstance(obj, (torch.device, torch.dtype)):
        return str(obj)
    if hasattr(obj, "__dict__"):
        return obj.__dict__
    return str(obj)


def init_logs(args):
    """Init log files. Mostly legacy behavior from DFME codebase."""
    os.makedirs(args.log_dir, exist_ok=True)

    # Save JSON with parameters
    with open(args.log_dir + "/parameters.json", "w") as f:
        json.dump(vars(args), f, default=_json_default)

    init_or_append_to_log_file(args.log_dir, "loss.csv", "epoch,loss_G,loss_S")
    init_or_append_to_log_file(args.log_dir, "accuracy.csv", "epoch,accuracy")
    init_or_append_to_log_file(os.getcwd(), "latest_experiments.txt",
                               args.experiment_name + ":" + args.log_dir, mode="a")

    if args.rec_grad_norm:
        init_or_append_to_log_file(args.log_dir, "norm_grad.csv", "epoch,G_grad_norm,S_grad_norm,grad_wrt_X")


def main():
    # Parse command line arguments and set certain variables based on those
    # args = parse_args(set_total_query_budget=True)
    # Print log directory

    import argparse
    parser = argparse.ArgumentParser()

    # 直接设置固定参数
    fixed_params = {
        'experiment_type': 'disguide',
        'epoch_itrs': 150,
        'log_interval': 30,
        'd_iter': 1,
        'grayscale': 8,
        'rep_iter': 3,
        'replay_size': 1000000,
        'input_space': 'pre-transform',
        'lambda_div': 0.2,
        'model': 'resnet34_8x',
        'steps': [0.4, 0.8],
        'ckpt': 'checkpoint/teacher/cifar10-resnet34_8x.pt',
        'dataset': 'cifar10',
        'lr_S': 0.03,
        'replay': 'Classic',
        'device': 1,  # GPU编号
        'query_budget': 20000000,  # 20M
        'log_dir': 'save_results/cifar10',
        'lr_G': 1e-4,
        'ensemble_size': 2,
        'batch_size': 256,
        'loss': 'hl',  # 或 'hl'、'l1'
        'student_model': 'ensemble_resnet18_8x',
        'scale': 0.3,
        'scheduler': 'multistep',
        'seed': 74919,
        # 创新点参数
        'use_boundary_sampling': True,
        'boundary_sample_ratio': 0.4,
        'use_mc_uncertainty': True,
        'mc_samples': 5,
        'mc_threshold': 0.6,
         # 先验：务必非零权重并提供组合权重
        'use_informativeness_prior': False,
        'informativeness_weight': 0.2,              # 先验强度，可按需调
        'informativeness_schedule': 'none',         # 或 linear
        'informativeness_warmup': 1,
        'informativeness_objective': 'mean',        # 或 topk_mean
        'informativeness_topk_ratio': None,         # 若用 topk_mean 再填
        'selector_percentiles': [0.5, 0.9],

        #以下2选1
        'informativeness_combine_strategy': 'weighted_sum', #或者 gate_then_rank
        'informativeness_weight_epistemic': 0.5,    # 必填：epistemic 权重
        'informativeness_weight_boundary': 0.5,     # 必填：boundary 权重
    
        # 'informativeness_combine_strategy': 'gate_then_rank',
        # 'informativeness_gate_quantile': 0.7,   # 例：保留 top 30% epistemic
        # 'informativeness_boundary_topk': 0.4,   # 例：在通过 gate 的样本里按 boundary 取前 40%

    
        # 是否让评分链路保留梯度
        'informativeness_differentiable': False,    # 需要保留梯度时改为 True
    }

    # 调用原 parse_args 但用固定参数覆盖
    args = parse_args(set_total_query_budget=True)
    for key, value in fixed_params.items():
        setattr(args, key, value)
    print(args.log_dir)

    # Build structured cfg objects for informativeness pipeline
    args.informativeness_cfg = _build_informativeness_cfg(args)
    args.selector_cfg = _build_selector_cfg(args)
    args.gen_cfg = _build_gen_cfg(args)
    args.gen_update_count = 0

    # Prepare the environment
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed(args.seed)
    np.random.seed(args.seed)
    random.seed(args.seed)
    torch.backends.cudnn.deterministic = args.cudnn_deterministic
    torch.backends.cudnn.benchmark = False

    # Load test_loader and transform functions based on CLI args. Ignore train_loader.
    _, test_loader, normalization = get_dataloader(args)
    print(f"\nLoaded {args.dataset} successfully.")
    # Display distribution information of dataset
    print_dataset_feature_distribution(data_loader=test_loader)

    # Initialize log files and directories
    init_logs(args)
    args.model_dir = f"{args.experiment_dir}/student_{args.model_id}"
    if not os.path.exists(args.model_dir):
        os.makedirs(args.model_dir)
    with open(f"{args.model_dir}/model_info.txt", "w") as f:
        json.dump(args.__dict__, f, indent=2, default=_json_default)
    config.log_file = open(f"{args.model_dir}/logs.txt", "w")

    # Set compute device
    use_cuda = torch.cuda.is_available() and not args.no_cuda
    args.device = torch.device("cuda:%d" % args.device if use_cuda else "cpu")
    print(f"Device is {args.device}")

    # Initialize tensorboard logger. Metrics are handled in metrics.py
    config.tboard_writer = SummaryWriter(f"tboard/general/{args.experiment_name}")

    args.normalization_coefs = None
    args.G_activation = torch.tanh

    # Set default number of classes. This will be moved to the CLI parser, eventually
    num_classes = 10 if args.dataset in ['cifar10', 'svhn', 'mnist'] else 100
    num_channels = 1 if args.dataset in ['mnist'] else 3
    args.num_classes = num_classes

    pprint(args, width=80)

    # Init teacher
    if args.model == 'resnet34_8x':
        teacher = network.resnet_8x.ResNet34_8x(num_classes=num_classes)

        args.ckpt = 'checkpoint/teacher/'+ args.dataset +'-resnet34_8x.pt'
        teacher.load_state_dict(torch.load(args.ckpt, map_location=args.device))
    elif args.model =='resnet18_8x':
        teacher = network.resnet_8x.ResNet18_8x(num_classes=num_classes)

        args.ckpt = 'checkpoint/teacher/'+ args.dataset +'-resnet18_8x.pt'
        teacher.load_state_dict(torch.load(args.ckpt, map_location=args.device))
    elif args.model == 'lenet5':
        teacher = network.lenet.LeNet5()

        args.ckpt = 'checkpoint/teacher/'+ args.dataset +'-lenet5.pt'
        teacher.load_state_dict(torch.load(args.ckpt, map_location=args.device))
    else:
        teacher = get_classifier(args.model, pretrained=True, num_classes=args.num_classes)

    # Wrap teacher model in a handler class together with the data transform
    teacher = TeacherModel(teacher, transform=normalization)
    teacher.eval()
    config.tboard_writer.add_graph(teacher, torch.rand((32, num_channels, 32, 32)))
    teacher = teacher.to(args.device)

    # Evaluate teacher on test dataset to verify accuracy is in expected range
    print_and_log("Teacher restored from %s" % (args.ckpt))
    print(f"\n\t\tTraining with {args.model} as a Target\n")
    accuracy, _ = get_model_accuracy_and_loss(args, teacher, test_loader, args.device)
    print('\nTeacher - Test set: Accuracy: {}/{} ({:.4f}%)\n'.format(np.round(accuracy * len(test_loader.dataset)),
                                                                     len(test_loader.dataset), accuracy))

    # Initialize a fresh student ensemble. Ensemble may be of size 1.
    student = get_classifier(args.student_model, pretrained=False, num_classes=args.num_classes,
                             ensemble_size=args.ensemble_size)
    for i in range(args.ensemble_size):
        config.tboard_writer.add_graph(student.get_model_by_idx(i), torch.rand((32, num_channels, 32, 32)))
    student = student.to(args.device)

    # Initialize generator
    generator = network.gan.GeneratorA(nz=args.nz, nc=num_channels, img_size=32, activation=args.G_activation,
                                       grayscale=args.grayscale)
    # config.tboard_writer.add_graph(generator, torch.rand((32, args.nz)))
    generator = generator.to(args.device)

    args.generator = generator
    args.student = student
    args.teacher = teacher

    # Compute theoretical query cost per training iteration. This will be compared with true value to verify correctness
    args.cost_per_iteration = compute_cost_per_iteration(args)
    number_epochs = args.query_budget // (args.cost_per_iteration * args.epoch_itrs) + 1

    print(f"\nTotal budget: {args.query_budget // 1000}k")
    print("Cost per iterations: ", args.cost_per_iteration)
    print("Total number of epochs: ", number_epochs)

    # Init optimizers for student and generator
    optimizer_S = optim.SGD(student.parameters(), lr=args.lr_S, weight_decay=args.weight_decay, momentum=0.9)
    optimizer_G = optim.Adam(generator.parameters(), lr=args.lr_G)

    # Compute learning rate drop iterations based on input percentages and iteration count.
    steps = sorted([int(step * number_epochs) for step in args.steps])
    print("Learning rate scheduling at steps: {}\n".format(steps))

    if args.scheduler == "multistep":
        scheduler_S = optim.lr_scheduler.MultiStepLR(optimizer_S, steps, args.scale)
        scheduler_G = optim.lr_scheduler.MultiStepLR(optimizer_G, steps, args.scale)
    elif args.scheduler == "cosine":
        scheduler_S = optim.lr_scheduler.CosineAnnealingLR(optimizer_S, number_epochs)
        scheduler_G = optim.lr_scheduler.CosineAnnealingLR(optimizer_G, number_epochs)

    best_acc = 0
    acc_list = []
    replay_memory = init_replay_memory(args)
    teacher_test_preds, _ = get_model_preds_and_true_labels(teacher, test_loader, args.device)

    # Accuracy milestones to log
    accuracy_goals = {0.75: 0, 0.8: 0, 0.85: 0, 0.9: 0}

    # Outer training loop.
    for epoch in range(1, number_epochs + 1):
        print_and_log(f"{args.experiment_name} epoch {epoch}")

        config.tboard_writer.add_scalar('Param/student_learning_rate', scheduler_S.get_last_lr()[0], args.current_query_count)
        config.tboard_writer.add_scalar('Param/generator_learning_rate', scheduler_G.get_last_lr()[0],
                                 args.current_query_count)

        # Inner training loop call
        student_preds, teacher_preds = train_epoch_ensemble(args, generator, student, teacher, args.device,
                                                            optimizer_S, optimizer_G, epoch, replay_memory)
        if args.scheduler != "none":
            scheduler_S.step()
            scheduler_G.step()
        replay_memory.new_epoch()

        # Test and log
        acc = log_test_metrics(student, test_loader, teacher_test_preds, args.device, args)
        for goal in accuracy_goals:
            if accuracy_goals[goal] == 0 and acc > goal:
                accuracy_goals[goal] = args.current_query_count / 1000000.0
                print_and_log(f"Reached {goal} accuracy goal in {accuracy_goals[goal]}m queries")

        eval_and_log_validation_metrics(config.tboard_writer, student_preds, teacher_preds, args)
        log_generator_distribution(config.tboard_writer, teacher_preds, args)

        acc_list.append(acc)
        # Store models
        if acc > best_acc:
            if not os.path.exists(f"{args.experiment_dir}/model_checkpoints"):
                os.makedirs(f"{args.experiment_dir}/model_checkpoints")
            best_acc = acc
            torch.save(student.state_dict(), f"{args.experiment_dir}/model_checkpoints/student_best.pt")
            torch.save(generator.state_dict(), f"{args.experiment_dir}/model_checkpoints/generator_best.pt")
        if epoch % args.generator_store_frequency == 0:
            if not os.path.exists(f"{args.experiment_dir}/model_checkpoints/generators"):
                os.makedirs(f"{args.experiment_dir}/model_checkpoints/generators")
            torch.save(generator.state_dict(),
                       f"{args.experiment_dir}/model_checkpoints/generators/generator_{args.current_query_count}.pt")
    log_hparams(config.tboard_writer, args)
    for goal in sorted(accuracy_goals):
        print_and_log(f"Goal {goal}: {accuracy_goals[goal]}")
    print_and_log("Best Acc=%.6f" % best_acc)


if __name__ == '__main__':
    print("torch version", torch.__version__)
    main()
