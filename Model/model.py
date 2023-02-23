import sys, os

sys.path.append(os.getcwd())
from Process.process import *
import torch as th
import torch.nn.functional as F
from torch_scatter import scatter_mean

from Process.rand5fold import *
from tools.evaluate import *
from torch_geometric.nn import GINConv, global_mean_pool
from torch.nn import Sequential, Linear, ReLU
import copy
import math
import torch.distributions as dist
import random
from torch.distributions import Normal
import torch.nn as nn
from torch_geometric.utils import subgraph

device = th.device('cuda:0' if th.cuda.is_available() else 'cpu')
class PriorDiscriminator(th.nn.Module):
    def __init__(self, input_dim):
        super().__init__()
        self.l0 = th.nn.Linear(input_dim, input_dim)
        self.l1 = th.nn.Linear(input_dim, input_dim)
        self.l2 = th.nn.Linear(input_dim, 1)

    def forward(self, x):
        h = F.relu(self.l0(x))
        h = F.relu(self.l1(h))
        return th.sigmoid(self.l2(h))


class FF(th.nn.Module):
    def __init__(self, input_dim):
        super().__init__()
        self.block = th.nn.Sequential(
            th.nn.Linear(input_dim, input_dim),
            th.nn.ReLU(),
            th.nn.Linear(input_dim, input_dim),
            th.nn.ReLU(),
            th.nn.Linear(input_dim, input_dim),
            th.nn.ReLU()
        )
        self.linear_shortcut = th.nn.Linear(input_dim, input_dim)

    def forward(self, x):
        return self.block(x) + self.linear_shortcut(x)


class Encoder(th.nn.Module):
    def __init__(self, num_features, dim, num_gc_layers):
        super(Encoder, self).__init__()

        self.num_gc_layers = num_gc_layers

        self.convs = th.nn.ModuleList()
        self.bns = th.nn.ModuleList()

        for i in range(num_gc_layers):
            if i:
                nn = Sequential(Linear(dim, dim), ReLU(), Linear(dim, dim))
            else:
                nn = Sequential(Linear(num_features, dim), ReLU(), Linear(dim, dim))
            conv = GINConv(nn)
            bn = th.nn.BatchNorm1d(dim)

            self.convs.append(conv)
            self.bns.append(bn)

    def forward(self, x, edge_index, batch):

        x_one = copy.deepcopy(x)
        xs_one = []
        for i in range(self.num_gc_layers):
            x_one = F.relu(self.convs[i](x_one, edge_index))
            xs_one.append(x_one)

        xpool_one = [global_mean_pool(x_one, batch) for x_one in xs_one]
        x_one = th.cat(xpool_one, 1)
        return x_one, th.cat(xs_one, 1)

    def get_embeddings(self, data):

        with th.no_grad():
            x, edge_index, batch = data.x, data.edge_index, data.batch
            graph_embed, node_embed = self.forward(x, edge_index, batch)
        return node_embed





def entro(t):
    epsilon = 1e-5  # to avoid log(0)
    probs = F.softmax(t, dim=-1)
    log_probs = th.log(probs + epsilon)
    H = -th.sum(probs * log_probs, dim=-1)
    return H


class Net(th.nn.Module):
    def __init__(self, hidden_dim, num_gc_layers, alpha=0.5, beta=1., gamma=.1):
        super(Net, self).__init__()

        self.alpha = alpha
        self.beta = beta
        self.gamma = gamma

        self.embedding_dim = mi_units = hidden_dim * num_gc_layers
        self.encoder = Encoder(5000, hidden_dim, num_gc_layers)

        self.local_d = FF(self.embedding_dim)  # Feed forward layer
        self.global_d = FF(self.embedding_dim)  # Feed forward layer
        self.mask_rate = nn.Parameter(th.zeros(1))  # learnable parameter for masking nodes
        self.edge_mask = nn.Parameter(th.zeros(1))
        self.nn = nn.Sequential(
            nn.Linear(in_features=5000, out_features=64),
            nn.ReLU(),
            nn.Linear(in_features=64, out_features=4)
        )
        self.init_emb()


    def generate_drop_edge(self,x, edgeindex):
        Z =self.nn(x)
        pi = th.sigmoid(th.matmul(Z, th.t(Z)))
        row = list(edgeindex[0])
        col = list(edgeindex[1])
        edgeindex_ib = []
        for i in range(len(row)):
            u, v = row[i], col[i]
            if th.distributions.Bernoulli(pi[u, v]).sample() == 1:
                edgeindex_ib.append(i)
        row_ib = [row[i] for i in edgeindex_ib]
        col_ib = [col[i] for i in edgeindex_ib]
        drop_edgeindex = [row_ib, col_ib]
        return th.LongTensor(drop_edgeindex).cuda()
    def init_emb(self):
        initrange = -1.5 / self.embedding_dim
        for m in self.modules():
            if isinstance(m, th.nn.Linear):
                th.nn.init.xavier_uniform_(m.weight.data)
                if m.bias is not None:
                    m.bias.data.fill_(0.0)

    def forward(self, data):

        x, edge_index, dropped_edge_index, batch, num_graphs, mask = data.x, data.edge_index, data.dropped_edge_index, data.batch, max(
            data.batch) + 1, data.mask
        x_pos_one=generate_mask_node(x)
        x_pos_two=generate_mask_node(x)
        dropped_edge_one=self.generate_drop_edge(x,edge_index)
        dropped_edge_two=self.generate_drop_edge(x,edge_index)

        _, M = self.encoder(x, edge_index, batch)

        y_pos_one, _ = self.encoder(x_pos_one, dropped_edge_one, batch)  #
        y_pos_two,_ = self.encoder(x_pos_two, dropped_edge_two, batch)  # drop edge
        # the encoded node feature matrix, corresponds to X in the L_ssl equation
        l_enc = self.local_d(M)  # feed forward
        # Compute representation for each augmented graph
        g_enc_pos = self.global_d(y_pos_one)
        g_enc_neg = self.global_d(y_pos_two)
        #compute average representation for graph pairs
        g_enc=(g_enc_pos + g_enc_neg) / 2
        IB_loss =ib_loss(l_enc,g_enc_pos,g_enc_neg,g_enc)



        return IB_loss
def ib_loss(l_enc, g_enc_pos, g_enc_neg, g_enc, beta=0.2, temperature=0.1, num_samples=2):
    # Compute contrastive loss
    margin = 2.0

    distance = th.sqrt(th.sum((g_enc_pos - g_enc_neg) ** 2))
    loss_cl = (1 - 1) * 0.5 * distance ** 2 + 1 * 0.5 * max(0, margin - distance) ** 2
    mean_H_IB = th.zeros(g_enc.shape[1]).cuda()
    var_H_IB = th.ones(g_enc.shape[1]).cuda()
    p_H_IB = dist.MultivariateNormal(mean_H_IB, th.diag(var_H_IB))

    # Compute KL divergence
    mean_H_IB_given_X = g_enc.mean(dim=0).cuda()
    var_H_IB_given_X = F.softplus(g_enc.var(dim=0)).cuda()
    p_H_IB_given_X = dist.MultivariateNormal(mean_H_IB_given_X, th.diag(var_H_IB_given_X))

    KL_loss = dist.kl_divergence(p_H_IB_given_X, p_H_IB).div(math.log(2))/128

    return loss_cl+0.2*KL_loss
def kl_divergence(p, q):
    return th.sum(p * th.log(p / q))
def generate_mask_node(x):
    # Generate the binary mask tensor with a certain probability (e.g., 0.6)
    d = x.shape[1]  # number of features
    mask_prob = 0.6
    mask = th.zeros(d).bernoulli_(mask_prob).cuda()

    # Randomly sample a feature vector from the data matrix
    sample_indices = th.randint(x.shape[0], size=(1,)).cuda()
    X_r = x[sample_indices].cuda()

    # Generate the masked feature matrix
    X_IB = (X_r + (x - X_r) * mask).cuda()
    return  X_IB
class Classfier(th.nn.Module):
    def __init__(self, in_feats, hid_feats, num_classes):
        super(Classfier, self).__init__()
        self.linear_one = BayesianLinear(5000 * 2, 2 * hid_feats)
        self.linear_two = BayesianLinear(2 * hid_feats, hid_feats)
        self.linear_three = BayesianLinear(in_feats, hid_feats)

        self.linear_transform = BayesianLinear(hid_feats * 2, 4)
        self.prelu = th.nn.PReLU()
        self.num_classes=num_classes
        self.uncertainty_weight=0.2
        for m in self.modules():
            self.weights_init(m)

    def weights_init(self, m):
        if isinstance(m, th.nn.Linear):
            th.nn.init.xavier_uniform_(m.weight.data)
            if m.bias is not None:
                m.bias.data.fill_(0.0)

    def forward(self, embed, data):
        ori = scatter_mean(data.x, data.batch, dim=0)
        root = data.x[data.rootindex]
        ori = th.cat((ori, root), dim=1)
        ori = self.linear_one(ori)
        ori = F.dropout(input=ori, p=0.5, training=self.training)
        ori = self.prelu(ori)
        ori = self.linear_two(ori)
        ori = F.dropout(input=ori, p=0.5, training=self.training)
        ori = self.prelu(ori)

        x = scatter_mean(embed, data.batch, dim=0)
        x = self.linear_three(x)
        x = F.dropout(input=x, p=0.5, training=self.training)
        x = self.prelu(x)

        out = th.cat((x, ori), dim=1)
        out = self.linear_transform(out)
        # x = F.log_softmax(out, dim=1)
        # pred_prob = F.softmax(x, dim=1)
        #make multiple predictions
        pred_probs = []
        for i in range(10):
            x = F.log_softmax(out, dim=1)
            pred_prob = F.softmax(x, dim=1)
            pred_probs.append(pred_prob)
        mean_pred_prob = th.stack(pred_probs).mean(dim=0)
        x = th.log(mean_pred_prob)
        kl_div = 0.0
        #calculate KL
        for module in self.modules():
            if hasattr(module, 'kl_loss'):
                kl_div += module.kl_loss()
        return x, pred_prob,kl_div
class BayesianLinear(th.nn.Module):
    def __init__(self, in_features, out_features):
        super(BayesianLinear, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.weights_mean = th.nn.Parameter(th.Tensor(out_features, in_features))
        self.weights_log_var = th.nn.Parameter(th.Tensor(out_features, in_features))
        self.bias_mean = th.nn.Parameter(th.Tensor(out_features))
        self.bias_log_var = th.nn.Parameter(th.Tensor(out_features))
        self.reset_parameters()

    def reset_parameters(self):
        th.nn.init.kaiming_uniform_(self.weights_mean, nonlinearity='relu')
        th.nn.init.constant_(self.weights_log_var, -10)
        th.nn.init.constant_(self.bias_mean, 0)
        th.nn.init.constant_(self.bias_log_var, -10)

    def forward(self, x):
        weights = dist.Normal(self.weights_mean, self.weights_log_var.exp().sqrt()).rsample()
        bias = dist.Normal(self.bias_mean, self.bias_log_var.exp().sqrt()).rsample()
        return F.linear(x, weights, bias)