import numpy as np
import pickle, os, argparse, sys, subprocess

import torch
import torch.nn as nn
import torch.nn.functional as F

def install(package):
    subprocess.call([sys.executable, "-m", "pip", "install", package])

install('dgl')

import dgl
from dgl.nn import GraphConv

class GCN(nn.Module):
    def __init__(self, in_feats, h_feats, num_classes):
        super(GCN, self).__init__()
        self.conv1 = GraphConv(in_feats, h_feats)
        self.conv2 = GraphConv(h_feats, num_classes)

    def forward(self, g, in_feat):
        h = self.conv1(g, in_feat)
        h = F.relu(h)
        h = self.conv2(g, h)
        return h

def build_karate_club_graph(edges):
    g = dgl.DGLGraph()
    g.add_nodes(node_count)
    src, dst = zip(*edges)
    src = np.asarray(src).astype('int64')
    dst = np.asarray(dst).astype('int64')
    g.add_edges(torch.tensor(src), torch.tensor(dst))
    # edges are directional in DGL; make them bidirectional
    g.add_edges(torch.tensor(dst), torch.tensor(src))
    return g

if __name__ == '__main__':
    
    parser = argparse.ArgumentParser()
    parser.add_argument('--epochs', type=int, default=30)
    parser.add_argument('--node_count', type=int)
    args, _    = parser.parse_known_args()
    epochs     = args.epochs
    node_count = args.node_count
    
    training_dir = os.environ['SM_CHANNEL_TRAINING']
    model_dir    = os.environ['SM_MODEL_DIR']

    # Load edges from pickle file
    with open(os.path.join(training_dir, 'edge_list.pickle'), 'rb') as f:
        edge_list = pickle.load(f)
    print(edge_list)
    
    G = build_karate_club_graph(edge_list)
    print('We have %d nodes.' % G.number_of_nodes())
    print('We have %d edges.' % G.number_of_edges())

    # The first layer transforms input features of size of 34 to a hidden size of 5.
    # The second layer transforms the hidden layer and produces output features of
    # size 2, corresponding to the two groups of the karate club.
    net = GCN(node_count, 5, 2)

    inputs = torch.eye(node_count)
    labeled_nodes = torch.tensor([0, node_count-1])  # only the instructor and the president nodes are labeled
    labels = torch.tensor([0,1])  # their labels are different

    optimizer = torch.optim.Adam(net.parameters(), lr=0.1)
    all_preds = []

    for epoch in range(epochs):
        preds = net(G, inputs)
        all_preds.append(preds)
        # we only compute loss for labeled nodes
        loss = F.cross_entropy(preds[labeled_nodes], labels)
        optimizer.zero_grad() # PyTorch accumulates gradients by default
        loss.backward() 
        optimizer.step()
        print('Epoch %d | Loss: %.4f' % (epoch, loss.item()))

    last_epoch = all_preds[epochs-1].detach().numpy()
    predicted_class = np.argmax(last_epoch, axis=-1)

    print(predicted_class)

    torch.save(net.state_dict(), os.path.join(model_dir, 'karate_club.pt'))

