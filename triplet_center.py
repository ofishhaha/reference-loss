import torch
import torch.nn as nn
import torch.nn.parallel
import torch.nn.functional as F
from torch.autograd import Variable
from torch.nn import Parameter
import numpy as np


class CenterLoss(nn.Module):
  def __init__(self, margin=0):
    super(CenterLoss, self).__init__()
    self.margin = margin
    self.ranking_loss_center = nn.MarginRankingLoss(margin=self.margin)
    self.centers = nn.Parameter(torch.randn(767, 2048)).cuda()  # for modelent40

    #self.centers = nn.Parameter(torch.randn(702, 2048)).cuda()  # for modelent40
    # self.centers = nn.Parameter(torch.randn(40, 40)) # for shapenet55

  def forward(self, inputs, targets):
    n = inputs.size(0)
    m = self.centers.size(0)
    dist = torch.pow(inputs, 2).sum(dim=1, keepdim=True).expand(n, m) + \
           torch.pow(self.centers, 2).sum(dim=1, keepdim=True).expand(m, n).t()
    dist.addmm_(1, -2, inputs, self.centers.t())
    dist = dist.clamp(min=1e-12).sqrt()

    # for each anchor, find the hardest positive and negative
    mask = torch.zeros(dist.size()).byte().cuda()
    for i in range(n):
      mask[i][targets[i].data] = 1

    # mask = targets.expand(n, n).eq(targets.expand(n, n).t())
    dist_ap, dist_an = [], []

    for i in range(n):
      dist_ap.append(dist[i][mask[i]].max())  # hardest positive center
      dist_an.append(dist[i][mask[i] == 0].min())  # hardest negative center
    #print(len(dist_ap))
    dist_ap = torch.stack(dist_ap)
    dist_an = torch.stack(dist_an)
    # generate a new label y
    # compute ranking hinge loss
    y = dist_an.data.new()
    y.resize_as_(dist_an.data)
    y.fill_(1)
    y = Variable(y)
    # y_i = 1, means dist_an > dist_ap + margin will casuse loss be zero
    loss = self.ranking_loss_center(dist_an, dist_ap, y)
    prec = (dist_an.data > dist_ap.data).sum() * 1. / y.size(0)

    # normalize data by batch size
    return loss