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

        self.encoder = encoder
        self.regressor = regressor

        self.kmeans = self.cluster_profiles(self.labels, k=opt.clusters)
        self.tau = opt.tau
        self.threshold = opt.confidence_threshold

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
        #D, I = kmeans.index.search(labels, 1)
        #C, N = np.unique(I, return_counts=True)

        #print(f"Cluster statistics: {[{int(c): n} for c, n in zip(C, N)]}")
        #print(f"Centroids: {kmeans.centroids}")
        return kmeans

    def snn(self, query, supports, targets):
        query = F.normalize(query)
        supports = F.normalize(supports)

        return (query @ supports.T / self.tau) @ targets

    def knn(self, query, supports, targets, k=5):
        query = F.normalize(query, dim=-1)
        supports = F.normalize(supports, dim=-1)

        dists = torch.cdist(query, supports)
        dists, indices = dists.topk(k, dim=-1, largest=False)

        # calculate weights based on distances
        weights = 1.0 / (dists + 1e-8)  # add a small value to prevent division by zero
        weights = weights / weights.sum(dim=-1, keepdim=True)  # normalize the weights

        target_values = torch.stack([targets[idx] for idx in indices])

        return (target_values * weights.unsqueeze(-1)).sum(dim=1)

    def consistency_loss(self, targets, anchor, pos, anchor_supports, pos_supports):
        targets1 = self.knn(anchor, anchor_supports, targets)
        with torch.no_grad():
            targets2 = self.knn(pos, pos_supports, targets)

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
