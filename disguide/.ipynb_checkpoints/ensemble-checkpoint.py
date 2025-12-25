# Helper class for having ensemble models.

import torch
import torch.nn as nn
import torch.nn.functional as F
import network


class Ensemble(nn.Module):
    """Wrapper modeule for an ensemble of clone models"""

    def __init__(self, ensemble_size: int = 16, num_classes: int = 10, student_model: str = "ensemble_resnet18_8x"):
        super(Ensemble, self).__init__()
        if student_model == 'ensemble_resnet18_8x':
            self.subnets = nn.ModuleList([network.resnet_8x.ResNet18_8x(num_classes) for i in range(ensemble_size)])
        elif student_model == 'ensemble_lenet5_half':
            self.subnets = nn.ModuleList([network.lenet.LeNet5Half() for i in range(ensemble_size)])
        else:
            raise NotImplementedError("Only supporting lenet5Half and Resnet18")
    
    def forward(self, x, idx: int = -1, **kwargs):
        """
        修改后的 forward，支持 **kwargs 透传 (例如 out_feature=True)
        """
        if idx >= 0:
            # 单个模型前向，支持 return features
            return self.subnets[idx].forward(x, **kwargs)
        
        # 整体前向 (通常只在 eval 或 metrics 时用，这里默认只返回 logits 以保持兼容性)
        # 如果你需要在 idx=-1 时也返回 features，逻辑会比较复杂，
        # 但在 train.py 中我们都是通过 idx 循环调用的，所以这里只改 idx>=0 即可满足训练需求。
        results = []
        for i in range(len(self.subnets)):
            results.append(self.subnets[i].forward(x, **kwargs))
            
        # 注意：如果 kwargs 含有 out_feature=True，results 里存的是 tuple (logits, feats)
        # 为了防止 crash，如果发现返回的是 tuple，我们只 stack logits 部分
        if isinstance(results[0], tuple):
            logits_only = [res[0] for res in results]
            return torch.stack(logits_only, dim=1)
            
        return torch.stack(results, dim=1)
    
    def variance(self, x):
        results = []
        with torch.no_grad():
            for i in range(len(self.subnets)):
                results.append(self.subnets[i].forward(x))
            return torch.var(F.softmax(torch.stack(results, dim=1), dim=-1), dim=1)
    
    def size(self):
        return len(self.subnets)
    
    def get_model_by_idx(self, idx):
        return self.subnets[idx]


    def enable_dropout(self):
        """启用所有Dropout层用于MC Dropout"""
        for subnet in self.subnets:
            for module in subnet.modules():
                if isinstance(module, (nn.Dropout, nn.Dropout2d)):
                    module.train()


    def get_epistemic_uncertainty(self, x, mc_samples=5):
        """计算MC Dropout认知不确定性"""
        all_uncertainties = []

        for idx in range(self.size()):
            model = self.subnets[idx]
            original_training = model.training
            model.eval()
            self.enable_dropout()

            mc_preds = []
            with torch.no_grad():
                for _ in range(mc_samples):
                    logits = model(x)
                    probs = F.softmax(logits, dim=-1)
                    mc_preds.append(probs)

            model.train(original_training)
            mc_preds = torch.stack(mc_preds, dim=0)
            epistemic = torch.var(mc_preds, dim=0).mean(dim=-1)
            all_uncertainties.append(epistemic)

        return torch.stack(all_uncertainties, dim=0).mean(dim=0)


    def get_boundary_score(self, x):
        """计算边界得分"""
        with torch.no_grad():
            predictions = []
            for idx in range(self.size()):
                logits = self.subnets[idx](x)
                probs = F.softmax(logits, dim=-1)
                predictions.append(probs)

            predictions = torch.stack(predictions, dim=0)
            max_probs = torch.max(predictions, dim=-1)[0]
            avg_confidence = max_probs.mean(dim=0)
            disagreement = torch.std(predictions, dim=0).sum(dim=-1)
            boundary_score = (1 - avg_confidence) * disagreement

        return boundary_score
