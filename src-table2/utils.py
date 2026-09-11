"""
utility functions
"""
import os

import scipy
from scipy.stats import sem
import numpy as np
from torch_scatter import scatter_add
from torch_geometric.utils import add_remaining_self_loops
from torch_geometric.utils.num_nodes import maybe_num_nodes
from torch_geometric.utils.convert import to_scipy_sparse_matrix
from sklearn.preprocessing import normalize
from torch_geometric.nn.conv.gcn_conv import gcn_norm
from torch_geometric.utils import to_undirected, k_hop_subgraph
from torch_sparse import SparseTensor
import networkx as nx
from networkx.algorithms.shortest_paths.generic import shortest_path_length

ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
import random



class MaxNFEException(Exception): pass


def set_rand_seed(rand_seed):
    rand_seed = rand_seed if rand_seed >= 0 else torch.initial_seed() % 4294967295  # 2^32-1
    random.seed(rand_seed)
    torch.manual_seed(rand_seed)
    torch.cuda.manual_seed_all(rand_seed)
    np.random.seed(rand_seed)



def feat_normalize(features, norm=None, lim_min=-1.0, lim_max=1.0):
    r"""
    Description
    -----------
    Feature normalization function.

    Parameters
    ----------
    features : torch.FloatTensor
        Features in form of ``N * D`` torch float tensor.
    norm : str, optional
        Type of normalization. Choose from ``["linearize", "arctan", "tanh", "standarize"]``.
        Default: ``None``.
    lim_min : float
        Minimum limit of feature value. Default: ``-1.0``.
    lim_max : float
        Minimum limit of feature value. Default: ``1.0``.

    Returns
    -------
    features : torch.FloatTensor
        Normalized features in form of ``N * D`` torch float tensor.

    """
    if norm == "linearize":
        k = (lim_max - lim_min) / (features.max() - features.min())
        features = lim_min + k * (features - features.min())
    elif norm == "arctan":
        features = (features - features.mean()) / features.std()
        features = 2 * np.arctan(features) / np.pi
    elif norm == "tanh":
        features = (features - features.mean()) / features.std()
        features = np.tanh(features)
    elif norm == "standardize":
        features = (features - features.mean()) / features.std()
    else:
        features = features

    return features


def get_index_induc(index_a, index_b):
    r"""

    Description
    -----------
    Get index under the inductive training setting.

    Parameters
    ----------
    index_a : tuple
        Tuple of index.
    index_b : tuple
        Tuple of index.

    Returns
    -------
    index_a_new : tuple
        Tuple of mapped index.
    index_b_new : tuple
        Tuple of mapped index.

    """

    i_a, i_b = 0, 0
    l_a, l_b = len(index_a), len(index_b)
    i_new = 0
    index_a_new, index_b_new = [], []
    while i_new < l_a + l_b:
        if i_a == l_a:
            while i_b < l_b:
                i_b += 1
                index_b_new.append(i_new)
                i_new += 1
            continue
        elif i_b == l_b:
            while i_a < l_a:
                i_a += 1
                index_a_new.append(i_new)
                i_new += 1
            continue
        if index_a[i_a] < index_b[i_b]:
            i_a += 1
            index_a_new.append(i_new)
            i_new += 1
        else:
            i_b += 1
            index_b_new.append(i_new)
            i_new += 1

    return index_a_new, index_b_new

def inductive_split(adj, split_idx):
    """
    inductive split adjs for PyG graphs
    will automatically use relative ids for splitted graphs
    """
    # adj =adj.to('cpu')
    adj_train = adj[split_idx["train"]][:,split_idx["train"]]
    train_mask = torch.zeros(adj.size(0)).bool()
    train_mask[split_idx["train"]] = 1
    val_mask = torch.zeros(adj.size(0)).bool()
    val_mask[split_idx["valid"]] = 1
    train_val_mask = torch.logical_or(train_mask, val_mask)
    train_val_mask = train_val_mask.to(adj.device())
    adj_val = adj[train_val_mask][:,train_val_mask]
   
    adj_test = adj
    return adj_train, adj_val, adj_test


def prune_graph(adj_test, target_idx, k):
    adj_test = adj_test.cpu()
    u, v, _ = adj_test.coo()
    _, edge_index, __, ___ = k_hop_subgraph(target_idx,k,torch.stack((u,v),dim=0))
    graph_size = torch.Size((adj_test.size(0),adj_test.size(1)))
    new_adj_test = SparseTensor(row=edge_index[1], col=edge_index[0], value=None, sparse_sizes=graph_size,is_sorted=True).to_symmetric()
    return new_adj_test


def target_select(model,adj,features,labels,target_idx,num):
    # num highest margin
    # num lowest margin
    # 2num random
    # with torch.no_grad():
    #     pred = model(features,adj)[target_idx]
    #     pred_y = pred.argmax(-1)
    # correct_idx = labels[target_idx].view(-1)==pred_y.view(-1)
    # assert len(correct_idx) >= 4*num
    # pred_max = pred.max(-1)[0]
    # second_y =  pred
    # second_y[torch.arange(pred_y.size(0)),pred_y] = -1e9
    # margin = (pred_max-second_y.max(-1)[0])[correct_idx]
    # margin_max = margin.argsort()
    # random_ids = torch.randperm(len(margin)-2*num)[:2*num]
    # selected_ids = torch.cat((margin_max[:num],margin_max[-num:],margin_max[num:-num][random_ids]),dim=0)


    # sanity check
    with torch.no_grad():
        pred = model(features,adj)[target_idx]
        pred_y = pred.argmax(-1)
    pred_sort, _ = pred.sort(-1,descending=True)
    correct_idx = labels[target_idx].view(-1)==pred_y.view(-1)
    print(f"Correctly classified nodes: {correct_idx.sum()}")
    new_margin = pred_sort[correct_idx,0]-pred_sort[correct_idx,1]
    new_margin_max = new_margin.argsort()
    random_ids = torch.randperm(len(new_margin)-2*num)[:2*num]
    # (min, max, random, random)
    selected_ids = torch.cat((new_margin_max[:num],new_margin_max[-num:],new_margin_max[num:-num][random_ids]),dim=0)

    # assert (new_margin_max[:num]!=margin_max[:num]).sum()==0, print((new_margin_max[:num]!=margin_max[:num]).sum()) 
    # assert (new_margin_max[-num:]!=margin_max[-num:]).sum()==0, print((new_margin_max[:num]!=margin_max[:num]).sum()) 

    return target_idx[selected_ids]



def rms_norm(tensor):
  return tensor.pow(2).mean().sqrt()


def make_norm(state):
  if isinstance(state, tuple):
    state = state[0]
  state_size = state.numel()

  def norm(aug_state):
    y = aug_state[1:1 + state_size]
    adj_y = aug_state[1 + state_size:1 + 2 * state_size]
    return max(rms_norm(y), rms_norm(adj_y))

  return norm


def print_model_params(model):
  total_num_params = 0
  print(model)
  for name, param in model.named_parameters():
    if param.requires_grad:
      print(name)
      print(param.data.shape)
      total_num_params += param.numel()
  print("Model has a total of {} params".format(total_num_params))


def adjust_learning_rate(optimizer, lr, epoch, burnin=50):
  if epoch <= burnin:
    for param_group in optimizer.param_groups:
      param_group["lr"] = lr * epoch / burnin


def gcn_norm_fill_val(edge_index, edge_weight=None, fill_value=0., num_nodes=None, dtype=None):
  num_nodes = maybe_num_nodes(edge_index, num_nodes)

  if edge_weight is None:
    edge_weight = torch.ones((edge_index.size(1),), dtype=dtype,
                             device=edge_index.device)

  if not int(fill_value) == 0:
    edge_index, tmp_edge_weight = add_remaining_self_loops(
      edge_index, edge_weight, fill_value, num_nodes)
    assert tmp_edge_weight is not None
    edge_weight = tmp_edge_weight

  row, col = edge_index[0], edge_index[1]
  deg = scatter_add(edge_weight, col, dim=0, dim_size=num_nodes)
  deg_inv_sqrt = deg.pow_(-0.5)
  deg_inv_sqrt.masked_fill_(deg_inv_sqrt == float('inf'), 0)
  return edge_index, deg_inv_sqrt[row] * edge_weight * deg_inv_sqrt[col]


def coo2tensor(coo, device=None):
  indices = np.vstack((coo.row, coo.col))
  i = torch.LongTensor(indices)
  values = coo.data
  v = torch.FloatTensor(values)
  shape = coo.shape
  print('adjacency matrix generated with shape {}'.format(shape))
  # test
  return torch.sparse.FloatTensor(i, v, torch.Size(shape)).to(device)


def get_sym_adj(data, opt, improved=False):
  edge_index, edge_weight = gcn_norm(  # yapf: disable
    data.edge_index, data.edge_attr, data.num_nodes,
    improved, opt['self_loop_weight'] > 0, dtype=data.x.dtype)
  coo = to_scipy_sparse_matrix(edge_index, edge_weight)
  return coo2tensor(coo)


def get_rw_adj_old(data, opt):
  if opt['self_loop_weight'] > 0:
    edge_index, edge_weight = add_remaining_self_loops(data.edge_index, data.edge_attr,
                                                       fill_value=opt['self_loop_weight'])
  else:
    edge_index, edge_weight = data.edge_index, data.edge_attr
  coo = to_scipy_sparse_matrix(edge_index, edge_weight)
  normed_csc = normalize(coo, norm='l1', axis=0)
  return coo2tensor(normed_csc.tocoo())


def get_rw_adj(edge_index, edge_weight=None, norm_dim=1, fill_value=0., num_nodes=None, dtype=None): ##该函数为输入的图（边索引和边权重）构建基于节点度的随机游走邻接矩阵。
  num_nodes = maybe_num_nodes(edge_index, num_nodes)

  if edge_weight is None: ##初始化边权重
    edge_weight = torch.ones((edge_index.size(1),), dtype=dtype,
                             device=edge_index.device)

  if not fill_value == 0:
    edge_index, tmp_edge_weight = add_remaining_self_loops(
      edge_index, edge_weight, fill_value, num_nodes) ##处理自环
    assert tmp_edge_weight is not None
    edge_weight = tmp_edge_weight

  row, col = edge_index[0], edge_index[1]
  indices = row if norm_dim == 0 else col
  deg = scatter_add(edge_weight, indices, dim=0, dim_size=num_nodes)##计算节点度，用于计算每个节点的度数（连接边的总权重）
  deg_inv_sqrt = deg.pow_(-1)
  edge_weight = deg_inv_sqrt[indices] * edge_weight if norm_dim == 0 else edge_weight * deg_inv_sqrt[indices] ##标准化边权重
  return edge_index, edge_weight


def mean_confidence_interval(data, confidence=0.95):
  """
  As number of samples will be < 10 use t-test for the mean confidence intervals
  :param data: NDarray of metric means
  :param confidence: The desired confidence interval
  :return: Float confidence interval
  """
  if len(data) < 2:
    return 0
  a = 1.0 * np.array(data)
  n = len(a)
  _, se = np.mean(a), scipy.stats.sem(a)
  h = se * scipy.stats.t.ppf((1 + confidence) / 2., n - 1)
  return h


def sparse_dense_mul(s, d):
  i = s._indices()
  v = s._values()
  return torch.sparse.FloatTensor(i, v * d, s.size())


def get_sem(vec):
  """
  wrapper around the scipy standard error metric
  :param vec: List of metric means
  :return:
  """
  if len(vec) > 1:
    retval = sem(vec)
  else:
    retval = 0.
  return retval


def get_full_adjacency(num_nodes):
  # what is the format of the edge index?
  edge_index = torch.zeros((2, num_nodes ** 2),dtype=torch.long)
  for idx in range(num_nodes):
    edge_index[0][idx * num_nodes: (idx + 1) * num_nodes] = idx
    edge_index[1][idx * num_nodes: (idx + 1) * num_nodes] = torch.arange(0, num_nodes,dtype=torch.long)
  return edge_index



from typing import Optional
import torch
from torch import Tensor
from torch_scatter import scatter, segment_csr, gather_csr


# https://twitter.com/jon_barron/status/1387167648669048833?s=12
# @torch.jit.script
def squareplus(src: Tensor, index: Optional[Tensor], ptr: Optional[Tensor] = None,
               num_nodes: Optional[int] = None) -> Tensor:
  r"""Computes a sparsely evaluated softmax.
    Given a value tensor :attr:`src`, this function first groups the values
    along the first dimension based on the indices specified in :attr:`index`,
    and then proceeds to compute the softmax individually for each group.

    Args:
        src (Tensor): The source tensor.
        index (LongTensor): The indices of elements for applying the softmax.
        ptr (LongTensor, optional): If given, computes the softmax based on
            sorted inputs in CSR representation. (default: :obj:`None`)
        num_nodes (int, optional): The number of nodes, *i.e.*
            :obj:`max_val + 1` of :attr:`index`. (default: :obj:`None`)

    :rtype: :class:`Tensor`
    """
  out = src - src.max()
  # out = out.exp()
  out = (out + torch.sqrt(out ** 2 + 4)) / 2

  if ptr is not None:
    out_sum = gather_csr(segment_csr(out, ptr, reduce='sum'), ptr)
  elif index is not None:
    N = maybe_num_nodes(index, num_nodes)
    out_sum = scatter(out, index, dim=0, dim_size=N, reduce='sum')[index]
  else:
    raise NotImplementedError

  return out / (out_sum + 1e-16)


# Counter of forward and backward passes.
class Meter(object):

  def __init__(self):
    self.reset()

  def reset(self):
    self.val = None
    self.sum = 0
    self.cnt = 0

  def update(self, val):
    self.val = val
    self.sum += val
    self.cnt += 1

  def get_average(self):
    if self.cnt == 0:
      return 0
    return self.sum / self.cnt

  def get_value(self):
    return self.val


class DummyDataset(object):
  def __init__(self, data, num_classes):
    self.data = data
    self.num_classes = num_classes


class DummyData(object):
  def __init__(self, edge_index=None, edge_Attr=None, num_nodes=None):
    self.edge_index = edge_index
    self.edge_attr = edge_Attr
    self.num_nodes = num_nodes
