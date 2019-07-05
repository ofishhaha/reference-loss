from __future__ import absolute_import

import torch
from torch import nn
from torch.autograd import Variable
import random

class TripletLoss(nn.Module):
    def __init__(self, margin=0):
        super(TripletLoss, self).__init__()
        self.margin = margin
        self.ranking_loss = nn.MarginRankingLoss(margin=margin, reduce=False)
        self.ranking_loss2 = nn.MarginRankingLoss(margin=margin, reduce=False)

    def forward(self, inputs, targets):
        #print(inputs)
        n = inputs.size(0)
        # Compute pairwise distance, replace by the official when merged
        dist = torch.pow(inputs, 2).sum(dim=1, keepdim=True).expand(n, n)
        dist = dist + dist.t()
        dist.addmm_(1, -2, inputs, inputs.t())
        dist = dist.clamp(min=1e-12).sqrt()  # for numerical stability
        # For each anchor, find the hardest positive and negative
        mask = targets.expand(n, n).eq(targets.expand(n, n).t())
        dist_ap, dist_an, dist_an2, dist_an3 = [], [], [], []
        #print(dist)
        for i in range(n):
            ap = dist[i][mask[i]].max()
            an = dist[i][mask[i] == 0].min()
            dist_ap.append(ap)
            dist_an.append(an)
            ap_index = (dist[i] == ap)*mask[i]
            #an_index = (dist[i] == an)*(mask[i] == 0)
            ind = random.randint(0, 127)
            #print(ind)
            if (ap_index.sum() > 1):
                ap_num = ap_index.sum()
                for i in range(ap_index.size(0)):
                    if(ap_index[i] == 1):
                        ap_index[i] = 0
                        ap_num -= 1
                        if (ap_num == 1):
                            break

            #print(ap_index, an_index)
            an2 = dist[ap_index].squeeze()[ind]
            an3 = dist[i][ind]

            dist_an2.append(an2)
            dist_an3.append(an3)

        dist_ap = torch.stack(dist_ap)
        dist_an = torch.stack(dist_an)
        #print(dist_an2)
        dist_an2 = torch.stack(dist_an2)
        dist_an3 = torch.stack(dist_an3)
        # Compute ranking hinge loss
        y = dist_an.data.new()
        y.resize_as_(dist_an.data)
        y.fill_(1)
        y = Variable(y)

        loss = self.ranking_loss(dist_an, dist_ap, y)
        loss2 = self.ranking_loss2(dist_an2, dist_ap, y)
        loss3 = torch.abs(dist_an3 - dist_an2)
        #print(loss3)
        #loss_final = (loss+0.1*dist_ap).sum()/loss.size(0)
        loss_final = (loss + loss3).sum() / loss.size(0)

        return loss_final,loss.sum() / loss.size(0) ,loss3.sum() / loss.size(0)
