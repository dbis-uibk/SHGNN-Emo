from functools import reduce
import math
import numpy as np
import torch
from torch import nn
from tqdm import tqdm
import torch.nn.functional as F

import faiss

from model.encoder import *


class SHGR(nn.Module):
    def __init__(self, opt, labels, encoder, regressor):
        super(SHGR, self).__init__()
        self.opt = opt
        self.dim = opt.dim
        self.num_clusters = opt.clusters

        self.labels = labels

        self.tau = opt.tau
        self.threshold = opt.confidence_threshold
        self.softmax = nn.Softmax(dim=1)

        self.encoder = encoder
        self.regressor = regressor

        self.kmeans = self.cluster_profiles(self.labels, k=opt.clusters)
        self.confidence_threshold = opt.confidence_threshold

        #self.reset_parameters()

    def reset_parameters(self):
        self.encoder.reset_parameters()
        self.regressor.reset_parameters()

    def cluster_profiles(self, labels, k=16):
        # clustering with faiss
        d = labels.shape[1]
        kmeans = faiss.Kmeans(d, k, niter=100, verbose=False, gpu=False)
        kmeans.train(labels)

        # print statistics of clusters
        D, I = kmeans.index.search(labels, 1)
        C, N = np.unique(I, return_counts=True)

        #print(f"Cluster statistics: {[{int(c): n} for c, n in zip(C, N)]}")
        #print(f"Centroids: {kmeans.centroids}")
        return kmeans

    def snn(self, query, supports, targets):
        query = F.normalize(query)
        supports = F.normalize(supports)

        return self.softmax(query @ supports.T / self.tau) @ targets

    def consistency_loss(self, targets, anchor, pos, anchor_supports, pos_supports):
        targets1 = self.snn(anchor, anchor_supports, targets)
        with torch.no_grad():
            targets2 = self.snn(pos, pos_supports, targets)

            # check cluster assignments via faiss
            D, I = self.kmeans.index.search(targets2.cpu(), 1)
            # scale D between 0 and 1
            D = (D - D.min()) / (D.max() - D.min())
            mask = (D < self.threshold).squeeze()

            targets2 = targets2[mask]

        targets1 = targets1[mask]

        loss = F.mse_loss(targets1, targets2)

        # check nan
        if torch.isnan(loss):
            return torch.tensor(0.0).to(targets.device)

        return loss

    def forward(self, x, adj):
        h = self.encoder(x, adj.t())
        h = self.regressor(h)

        return h
