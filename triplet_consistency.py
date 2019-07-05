from __future__ import absolute_import

import torch
from torch import nn
from torch.autograd import Variable


class TripletLoss(nn.Module):
    def __init__(self, margin=0):
        super(TripletLoss, self).__init__()
        self.margin = margin
        self.ranking_loss = nn.MarginRankingLoss(margin=margin, reduce=False)
        self.ranking_loss2 = nn.MarginRankingLoss(margin=margin, reduce=False)

    def forward(self, inputs, targets):
        #print(inputs)
        n = inputs.size(0)
        dist = torch.pow(inputs, 2).sum(dim=1, keepdim=True).expand(n, n)
        dist = dist + dist.t()
        dist.addmm_(1, -2, inputs, inputs.t())
        dist = dist.clamp(min=1e-12).sqrt()  # for numerical stability
        # For each anchor, find the hardest positive and negative
        mask = targets.expand(n, n).eq(targets.expand(n, n).t())
        dist_ap, dist_an, dist_an2 = [], [], []

        for i in range(n):
            ap = dist[i][mask[i]].max()
            an = dist[i][mask[i] == 0].min()
            dist_ap.append(ap)
            dist_an.append(an)
            ap_index = (dist[i] == ap)*mask[i]
            an_index = (dist[i] == an)*(mask[i] == 0)

            if (ap_index.sum() > 1):
                ap_num = ap_index.sum()
                for i in range(ap_index.size(0)):
                    if(ap_index[i] == 1):
                        ap_index[i] = 0
                        ap_num -= 1
                        if (ap_num == 1):
                            break

            if (an_index.sum() > 1):
                an_num = an_index.sum()
                for i in range(an_index.size(0)):
                    if (an_index[i] == 1):
                        an_index[i] = 0
                        an_num -= 1
                        if (an_num == 1):
                            break

            an2 = dist[ap_index].squeeze()[an_index]
            dist_an2.append(an2)

        dist_ap = torch.stack(dist_ap)
        dist_an = torch.stack(dist_an)
        dist_an2 = torch.cat(dist_an2)

        y = dist_an.data.new()
        y.resize_as_(dist_an.data)
        y.fill_(1)
        y = Variable(y)

        lam = 1
        loss = self.ranking_loss(dist_an, dist_ap, y)
        loss2 = self.ranking_loss2(dist_an2, dist_ap, y)
        loss3 = torch.abs(dist_an - dist_an2)
        
        loss_final = (loss + lam*loss3).sum() / loss.size(0)
        l1 = loss.sum()/ loss.size(0)
        l2 = loss3.sum()/ loss.size(0)
        return loss_final, l1, l2
