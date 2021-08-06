import torch
import torch.nn as nn


def _accuracy_top1(outputs, targets):
    _, predict = torch.max(outputs.data, 1)
    correct = predict.eq(targets.data).cpu().sum().item()
    return correct / outputs.size(0)


def accuracy(output, target, topk=(1,)):
    if topk == 1 or topk == (1,):
        return _accuracy_top1(output, target)

    maxk = max(topk)
    batch_size = target.size(0)

    _, pred = output.topk(maxk, 1, True, True)
    pred = pred.t()
    correct = pred.eq(target.view(1, -1).expand_as(pred))

    res = []
    for k in topk:
        correct_k = correct[:k].reshape(-1).float().sum(0, keepdim=True)
        res.append(correct_k.div_(batch_size).item())
    return res


class ArchitectureResult:
    def __init__(self, architecture: dict, reward: float):
        self.architecture = architecture
        self.reward = reward


class LabelSmoothingLoss(nn.Module):
    def __init__(self, smoothing=0.1, dim=-1):
        super(LabelSmoothingLoss, self).__init__()
        self.confidence = 1.0 - smoothing
        self.smoothing = smoothing
        self.dim = dim

    def forward(self, pred, target):
        pred = pred.log_softmax(dim=self.dim)
        num_classes = pred.size(self.dim)
        with torch.no_grad():
            true_dist = torch.zeros_like(pred)
            true_dist.fill_(self.smoothing / (num_classes - 1))
            true_dist.scatter_(1, target.data.unsqueeze(1), self.confidence)
        return torch.mean(torch.sum(-true_dist * pred, dim=self.dim))
