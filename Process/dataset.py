import os
import random

import numpy as np
import torch
from Process.getTwittergraph import Node_tweet
from torch.utils.data import Dataset
from torch_geometric.data import Data
import torch.nn as nn

def collate_fn(data):
    return data

class BiGraphDataset(Dataset):
    def __init__(self, fold_x, treeDic,lower=2, upper=100000, tddroprate=0,budroprate=0,
                 data_path=os.path.join('..','..', 'data', 'Twitter16graph')):
        self.fold_x = list(filter(lambda id: id in treeDic and len(treeDic[id]) >= lower and len(treeDic[id]) <= upper, fold_x))
        self.treeDic = treeDic
        self.data_path = data_path
        self.tddroprate = tddroprate  # rate of dropedge
        self.budroprate = budroprate
        self.nn = nn.Sequential(
        nn.Linear(in_features= 5000, out_features= 64),
        nn.ReLU(),
        nn.Linear(in_features= 64, out_features =4)
        )
        self.mask_rate = nn.Parameter(torch.zeros(1))  # learnable parameter for masking nodes
        self.edge_mask = nn.Parameter(torch.zeros(1))
    def __len__(self):
        return len(self.fold_x)

    def __getitem__(self, index):
        id =self.fold_x[index]
        data=np.load(os.path.join(self.data_path, id + ".npz"), allow_pickle=True)  # load the graph
        edgeindex = data['edgeindex'] # edgeindex->2 x num_nodes
        X = torch.tensor(data['x'], dtype=torch.float32)
        Z = self.nn(X)
        # if self.tddroprate > 0: # utilize dropedge
        #     row = list(edgeindex[0])
        #     col = list(edgeindex[1])
        #     length = len(row)
        #     poslist = random.sample(range(length), int(length * (1 - self.tddroprate)))
        #     poslist = sorted(poslist)
        #     row = list(np.array(row)[poslist])
        #     col = list(np.array(col)[poslist])
        #     drop_edgeindex = [row, col]


        tree = self.treeDic[id]
        index2node = {}
        for i in tree:
            node = Node_tweet(idx=i)
            index2node[i] = node

        for j in tree:
            indexC = j
            indexP = tree[j]['parent']
            nodeC = index2node[indexC]
            ## not root node ##
            if not indexP == 'None':
                nodeP = index2node[int(indexP)]
                nodeC.parent = nodeP
                nodeP.children.append(nodeC)
            ## root node ##
            else:
                rootindex = indexC - 1
                root_index = nodeC.index
                root_word = nodeC.word
        node_pi = torch.sigmoid(self.mask_rate)  # node mask probability
        edge_pi = torch.sigmoid(self.edge_mask)  # edge mask probability
        edge_pi = torch.sigmoid(torch.matmul(Z, Z.t()))
        mask = (torch.rand(data['x'].shape[0]) < node_pi).to(torch.bool)  # mask node
        if self.tddroprate > 0:
            # drop edge
            pi = torch.sigmoid(torch.matmul(Z, torch.t(Z)))
            row = list(edgeindex[0])
            col = list(edgeindex[1])
            edgeindex_ib = []
            for i in range(len(row)):
                u, v = row[i], col[i]
                if torch.distributions.Bernoulli(pi[u, v]).sample() == 1:
                    edgeindex_ib.append(i)
            row_ib = [row[i] for i in edgeindex_ib]
            col_ib = [col[i] for i in edgeindex_ib]
            drop_edgeindex = [row_ib, col_ib]
        mask = [0 for _ in range(len(index2node))]
        root_node = index2node[int(rootindex + 1)]
        que = root_node.children.copy()
        noise = 0.1 * np.random.randn(*data['x'].shape)
        while len(que) > 0:
            cur = que.pop()
            if random.random() >= self.mask_rate:
                mask[int(cur.idx) - 1] = 1
                for child in cur.children:
                    que.append(child)
        mask[rootindex] = 0
        if self.tddroprate > 0:
            return Data(x=torch.tensor(data['x'],dtype=torch.float32), # x->num_nodes x features ; edge_index->2 x num_nodes(top down)
                        x_pos=torch.tensor(data['x_pos'], dtype=torch.float32),
                        mask = torch.tensor(mask, dtype=torch.bool),
                        edge_index=torch.LongTensor(edgeindex),#BU_edge_index=torch.LongTensor(bunew_edgeindex), # BU_edge_index->2 x num_nodes(bottom up))
                        dropped_edge_index=torch.LongTensor(drop_edgeindex),
                 y=torch.LongTensor([int(data['y'])]), root=torch.LongTensor(data['root']), # y->label
                 rootindex=torch.LongTensor([int(data['rootindex'])]))  # rootindex->the index of source tweet
        else:
            return Data(x=torch.tensor(data['x'], dtype=torch.float32),
                        # x->num_nodes x features ; edge_index->2 x num_nodes(top down)
                        x_pos=torch.tensor(data['x_pos'], dtype=torch.float32),
                        mask=torch.tensor(mask, dtype=torch.bool),
                        edge_index=torch.LongTensor(edgeindex),
                        y=torch.LongTensor([int(data['y'])]), root=torch.LongTensor(data['root']),  # y->label
                        rootindex=torch.LongTensor([int(data['rootindex'])]))  # rootindex->the index of source tweet
