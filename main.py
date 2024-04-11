import argparse

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler, StandardScaler
import scipy.sparse
import torch
from torch_sparse import SparseTensor

import torch_geometric
from torch_geometric.loader import NeighborSampler, NeighborLoader, GraphSAINTRandomWalkSampler, HGTLoader
from torch_geometric.data import DataLoader
from torch_geometric.data import Data, HeteroData
from torch_geometric.nn import to_hetero
from torch_geometric.utils.convert import from_scipy_sparse_matrix

from preprocessing.utils import *
from model.utils import *
import pickle
import csv

from model.shgr import SHGR
from model.trainer import Trainer
from model.encoder import *
from model.regressor import Regressor
from model.sampler import *

parser = argparse.ArgumentParser()
parser.add_argument('--dataset', '-d', default='sample', help='sample/music4all-onion/m4a')

parser.add_argument('--input-dim', type=int, default=50)
parser.add_argument('--dim', type=int, default=100)
parser.add_argument('--layers', type=int, default=2, help='the number of gnn layers')
parser.add_argument('--gnn-dropout', type=float, default=0.2)
parser.add_argument('--dropout', type=float, default=0.2)
parser.add_argument('--projection', type=int, default=1)

parser.add_argument('--neighbors', type=int, default=20)

parser.add_argument('--lr', type=float, default=0.001, help='learning rate')
parser.add_argument('--l2', type=float, default=1e-5, help='l2 penalty')

parser.add_argument("--df_1", type=float, default=0.2)
parser.add_argument("--de_1", type=float, default=0.2)
parser.add_argument("--df_2", type=float, default=0.1)
parser.add_argument("--de_2", type=float, default=0.1)

parser.add_argument('--clusters', type=int, default=8)
parser.add_argument('--confidence-threshold', '-ct', type=float, default=0.1)
parser.add_argument('--tau', '-t', type=float, default=0.1)

parser.add_argument('--alpha', '-a', type=float, default=0.5)
parser.add_argument('--beta', '-b', type=float, default=0.5)

parser.add_argument("--epochs", '-e', type=int, default=1000)
parser.add_argument("--batch-size", '-bs', type=int, default=100)
parser.add_argument('--patience', type=int, default=30)

parser.add_argument('--seed', type=int, default=2020)
parser.add_argument('--ratio', type=float, default=0.2)
parser.add_argument('--folds', type=int, default=10)

opt = parser.parse_args()


def main():
    opt.dataset = 'music4all-onion'
    opt.num_node = 109267
    opt.batch_size = 20000
    opt.neighbors = 40
    opt.folds = 5

    opt.layers = 1
    opt.de_1 = 0.35
    opt.df_1 = 0.15
    opt.de_2 = 0.2
    opt.df_2 = 0.15

    opt.gnn_dropout = 0.2
    opt.dropout = 0.3

    opt.alpha = 30  # unsup loss weight
    opt.beta = 0.2  # consistency loss weight

    opt.clusters = 8

    init_seed(opt.seed)
    print(opt)

    adj_sessions = scipy.sparse.load_npz('data/' + opt.dataset + '/adj_sessions.npz')
    adj_genres = scipy.sparse.load_npz('data/' + opt.dataset + '/adj_genres.npz')
    adj_tags = scipy.sparse.load_npz('data/' + opt.dataset + '/adj_tags.npz')

    features, label_dict = load_preprocessed_data('data/' + opt.dataset)
    y_node_ids = np.array(list(label_dict.keys()))
    y_labeled = np.array(list(label_dict.values()))

    y = torch.zeros((opt.num_node, 9))
    y[y_node_ids] = torch.FloatTensor(y_labeled)

    train_indexes, test_indexes = train_test_split(y_node_ids, test_size=opt.ratio, random_state=opt.seed)

    train_mask = torch.zeros(opt.num_node, dtype=torch.bool)
    test_mask = torch.zeros(opt.num_node, dtype=torch.bool)
    train_mask[train_indexes] = True
    test_mask[test_indexes] = True

    session_edge_index, session_edge_weight = from_scipy_sparse_matrix(adj_sessions)
    genre_edge_index, genre_edge_weight = from_scipy_sparse_matrix(adj_genres)
    tag_edge_index, tag_edge_weight = from_scipy_sparse_matrix(adj_tags)

    session_edge_index, session_edge_weight = normalize_edge_weights(session_edge_index, session_edge_weight,
                                                                     num_nodes=opt.num_node)
    genre_edge_index, genre_edge_weight = normalize_edge_weights(genre_edge_index, genre_edge_weight,
                                                                 num_nodes=opt.num_node)
    tag_edge_index, tag_edge_weight = normalize_edge_weights(tag_edge_index, tag_edge_weight, num_nodes=opt.num_node)

    data = HeteroData(
        track={
            'x': torch.FloatTensor(features),
            'y': y,
            'train_mask': train_mask,
            'test_mask': test_mask,
            'label_mask': train_mask | test_mask
        },
        track__session__track={
            'edge_index': session_edge_index,
            'edge_weight': session_edge_weight
        },
        track__tag__track={
            'edge_index': tag_edge_index,
            'edge_weight': tag_edge_weight
        },
        track__genre__track={
            'edge_index': genre_edge_index,
            'edge_weight': genre_edge_weight
        }
    )

    train_loader = NeighborLoader(data, num_neighbors=[opt.neighbors] * opt.layers,
                                  shuffle=True, batch_size=opt.batch_size,
                                  input_nodes=('track'))
    
    encoder = to_hetero(HeteroGraphSage(input_dim=opt.input_dim,
                                        dim=opt.dim,
                                        num_layers=opt.layers,
                                        dropout=opt.gnn_dropout,
                                        projection=opt.projection
                                        ),
                        data.metadata(), aggr='mean')

    regressor = Regressor(opt.dim if opt.layers > 0 or opt.projection else opt.input_dim, opt.dim, num_targets=9, dropout=opt.dropout)

    model = trans_to_cuda(SHGR(opt, y_labeled, encoder=encoder, regressor=regressor))
    print(model)

    trainer = Trainer(
        opt,
        model,
        data,
    )
    print(trainer)

    print('start training')
    #results = trainer.train(train_loader, opt.epochs)
    results = trainer.train_folds(data, y_labeled, folds=opt.folds)
    #results = trainer.tune(data, y_labeled)

    # formatted print of results
    for key, value in results.items():
        print(f'{key}: {value:.4f}')

    print('\n')
    print(opt)


if __name__ == '__main__':
    main()
