import torch
import torch.nn as nn

class BCEWithoutLogits(nn.Module):
    def __init__(self):
        super(BCEWithoutLogits, self).__init__()
        self.bce = torch.nn.BCEWithLogitsLoss()

    def BCEScore(self, input, tar):
        _, max_arg = torch.max(torch.cat([input, 1 - input], 1), 1)
        _, tar_arg = torch.max(torch.cat([tar, 1 - tar], 1), 1)
        diff = (max_arg - tar_arg).float()
        dist = torch.clamp(torch.abs(diff), 0, 1)
        score = 1 - dist
        return score.mean()

    def forward(self, src, tar):
        loss = self.bce(src, tar)
        score = self.BCEScore(src, tar)
        return {0: loss,
                'score': score,
                }
