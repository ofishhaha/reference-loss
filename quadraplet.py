from __future__ import absolute_import

import torch
from torch import nn
from torch.autograd import Variable


class TripletLoss(nn.Module):
    def __init__(self, margin=0):
        super(TripletLoss, self).__init__()
        self.margin = margin
        self.ranking_loss = nn.MarginRankingLoss(margin=margin, reduce=False)
        #self.ranking_loss2 = nn.MarginRankingLoss(margin=margin*1.3, reduce=False)

    def forward(self, inputs, targets):
        n = inputs.size(0)
        # Compute pairwise distance, replace by the official when merged
        dist = torch.pow(inputs, 2).sum(dim=1, keepdim=True).expand(n, n)
        dist = dist + dist.t()
        dist.addmm_(1, -2, inputs, inputs.t())
        dist = dist.clamp(min=1e-12).sqrt()  # for numerical stability
        # For each anchor, find the hardest positive and negative
        mask = targets.expand(n, n).eq(targets.expand(n, n).t())
        dist_ap, dist_an, dist_an2, dist_an3, dist_an4 = [], [], [], [], []

        for i in range(n):
            ap = dist[i][mask[i]].max()
            an = dist[i][mask[i] == 0].min()
            dist_ap.append(ap)
            dist_an.append(an)
            ap_index = (dist[i] == ap)*mask[i]
            an_index = (dist[i] == an)*(mask[i] == 0)

            if (ap_index.sum() > 1):
                ap_num = ap_index.sum()
                for j in range(ap_index.size(0)):
                    if(ap_index[j] == 1):
                        ap_index[j] = 0
                        ap_num -= 1
                        if (ap_num == 1):
                            break

            if (an_index.sum() > 1):
                an_num = an_index.sum()
                for j in range(an_index.size(0)):
                    if (an_index[j] == 1):
                        an_index[j] = 0
                        an_num -= 1
                        if (an_num == 1):
                            break

            #print(ap_index, an_index)
            an2 = dist[ap_index].squeeze()[an_index]
            dist_an2.append(an2)

            maskc = (mask[an_index].squeeze() == 0) - mask[i]

            ok_min = dist[an_index].squeeze()[maskc].min()
            ok_min2 = (dist[an_index].squeeze() == ok_min) * maskc

            if (ok_min2.sum() > 1):
                ok_num = ok_min2.sum()
                for j in range(ok_min2.size(0)):
                    if (ok_min2[j] == 1):
                        ok_min2[j] = 0
                        ok_num -= 1
                        if (ok_num == 1):
                            break

            an3 = dist[i].squeeze()[ok_min2].squeeze().squeeze()
            an4 = dist[ap_index].squeeze()[ok_min2].squeeze()
            dist_an3.append(an3)
            dist_an4.append(an4)

        dist_ap = torch.stack(dist_ap)
        dist_an = torch.stack(dist_an)
        #print(dist_an2)
        dist_an2 = torch.cat(dist_an2)
        dist_an3 = torch.stack(dist_an3)
        dist_an4 = torch.stack(dist_an4)

        # Compute ranking hinge loss
        y = dist_an.data.new()
        y.resize_as_(dist_an.data)
        y.fill_(1)
        y = Variable(y)

        loss = self.ranking_loss(dist_an, dist_ap, y)
        #loss2 = self.ranking_loss2(dist_an2, dist_ap, y)
        loss3 = torch.abs(dist_an - dist_an2)
        loss4 = torch.abs(dist_an3 - dist_an4)
        loss_final = (loss + loss3 + loss4).sum() / loss.size(0)
        #loss_final = (loss+loss2+loss3).sum()/loss.size(0)

        return loss_final
