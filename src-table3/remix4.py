import argparse
import time
import os
import numpy as np
import torch
#from torch_geometric.nn import GCNConv, ChebConv  # noqa
import torch.nn.functional as F
from ogb.nodeproppred import Evaluator
from GNN import GNN
from data import get_dataset, set_train_val_test_split
from best_params import best_params_dict
from utils import ROOT_DIR, target_select, prune_graph, inductive_split, get_index_induc
import sys
import json
from sklearn.metrics import roc_auc_score
from torch_geometric.utils import is_undirected, to_undirected
import random
from run_config import parser

from copy import deepcopy
from attacks.rnd import RND
from attacks.vanilla import Vanilla
from attacks.speit import SPEIT
from attacks.pgd import PGD
from attacks.gia import GIA
from attacks.seqgia import SEQGIA
from attacks.agia import AGIA
import torch_geometric.transforms as T
import torch_sparse
from torch_sparse import SparseTensor
from torch_geometric.data import Data
from torch_geometric.datasets import Planetoid
from ogb.nodeproppred import Evaluator, PygNodePropPredDataset

#from robust_diffusion.attacks import create_attack, PGD as PGD_ad

import os
os.environ['CUDA_LAUNCH_BLOCKING'] = '1'

torch.autograd.set_detect_anomaly(True)


@torch.no_grad()
def sep_test(model, x, adj_t, y, target_idx, evaluator):
    model.eval()
    out = model(x, adj_t) #

    out = out[target_idx] if target_idx.size(0) < out.size(0) else out
    y = y[target_idx] if out.size(0) < y.size(0) else y

    y_pred = out.argmax(dim=-1, keepdim=True)
    acc = evaluator.eval({
        'y_true': y.unsqueeze(-1),
        'y_pred': y_pred,
    })['acc']
    return acc


def eval_robustness(model, features, adj, target_idx, labels, device, args, run):
    # when evaluating robustness in blackbox setting
    # the attacked graph&data will be loaded from pre-defined path
    if args.eval_robo_blk:
        graph_path = os.path.join(args.save_attack,args.dataset)+f"_{args.eval_attack}"
        if args.eval_target:
            graph_path += "_target"
        if args.mul_run and run>0:
            graph_path+=f"_{run}"
        graph_path += ".pt"
        print(f"Load graph from: {graph_path}")
        new_data = torch.load(graph_path)
        new_data = T.ToSparseTensor()(new_data)
        feat_attack = new_data.x[new_data.y.size(0):].to(device)
        adj_attack = new_data.adj_t.to(device)
        if args.eval_target:
            target_idx = new_data.target_idx
        return feat_attack, adj_attack, target_idx

    # initialize the corresponding adversary
    if "speit" in args.eval_attack.lower():
        # multi-layer is the original proposal, 
        # but the attack perf is bad in small graphs 
        attacker = SPEIT(epsilon=args.attack_lr,
                   n_epoch=args.attack_epoch,
                   n_inject_max= args.n_inject_max,
                   n_edge_max= args.n_edge_max,
                   feat_lim_min=args.feat_lim_min,
                   feat_lim_max=args.feat_lim_max,
                   inject_mode="multi-layer" if "ml" in args.eval_attack.lower() else "random",
                   device=device,
                   early_stop=args.early_stop) 
    elif args.eval_attack.lower() == "gia":
        attacker = GIA(epsilon=args.attack_lr,
                 n_epoch=args.attack_epoch,
                 n_inject_max= args.n_inject_max,
                 n_edge_max= args.n_edge_max,
                 feat_lim_min=args.feat_lim_min,
                 feat_lim_max=args.feat_lim_max,
                 device=device,
                 early_stop=args.early_stop,
                 disguise_coe=args.disguise_coe,
                 hinge=args.hinge)
    elif args.eval_attack.lower() == "seqgia":
        attacker = SEQGIA(epsilon=args.attack_lr,
                 n_epoch=args.attack_epoch,
                 a_epoch=args.agia_epoch,
                 n_inject_max= args.n_inject_max,
                 n_edge_max= args.n_edge_max,
                 feat_lim_min=args.feat_lim_min,
                 feat_lim_max=args.feat_lim_max,
                 device=device,
                 early_stop=args.early_stop,
                 disguise_coe=args.disguise_coe,
                 sequential_step=args.sequential_step,
                 injection=args.injection,
                 feat_upd=args.feat_upd,
                 branching=args.branching,
                 iter_epoch=args.iter_epoch,
                 agia_pre=args.agia_pre,
                 hinge=args.hinge)
    elif args.eval_attack.lower() == "pgd":
        attacker = PGD(epsilon=args.attack_lr, #决定了每次更新对抗样本时的最大扰动幅度。
                 n_epoch=args.attack_epoch, # 控制 PGD 攻击的迭代次数。
                 n_inject_max= args.n_inject_max, #控制注入的对抗节点的最大数量。
                 n_edge_max= args.n_edge_max, #控制每个对抗节点可以连接的最大边数,每个对抗节点可以与图中的原始节点连接，形成新的边。
                 feat_lim_min=args.feat_lim_min, #控制对抗节点特征的最小值和最大值。
                 feat_lim_max=args.feat_lim_max,
                 device=device,
                 early_stop=args.early_stop) #控制是否启用早停机制。如果 early_stop=True，则在攻击过程中如果损失不再显著增加，攻击会提前停止。
    elif args.eval_attack.lower() in ["agia"]:
        attacker = AGIA(epsilon=args.attack_lr,
                 n_epoch=args.attack_epoch,
                 a_epoch=args.agia_epoch,
                 n_inject_max= args.n_inject_max,
                 n_edge_max= args.n_edge_max,
                 feat_lim_min=args.feat_lim_min,
                 feat_lim_max=args.feat_lim_max,
                 device=device,
                 early_stop=args.early_stop,
                 disguise_coe=args.disguise_coe,
                 opt=args.eval_attack.lower()[0],
                 iter_epoch=args.iter_epoch)
    elif args.eval_attack.lower() == "rnd":
        attacker = RND(epsilon=args.attack_lr,
                 n_epoch=args.attack_epoch,
                 n_inject_max= args.n_inject_max,
                 n_edge_max= args.n_edge_max,
                 feat_lim_min=args.feat_lim_min,
                 feat_lim_max=args.feat_lim_max,
                 device=device)
    else:
        attacker = Vanilla(epsilon=args.attack_lr,
                 n_epoch=args.attack_epoch,
                 n_inject_max= args.n_inject_max,
                 n_edge_max= args.n_edge_max,
                 feat_lim_min=args.feat_lim_min,
                 feat_lim_max=args.feat_lim_max,
                 device=device)
    
    attack_labels = labels if args.attack_label else None
    if args.eval_target:
        target_idx = target_select(model,adj,features,labels,target_idx,args.target_num)
    if args.prune_graph:
        new_adj_test = prune_graph(adj, target_idx, args.num_layers)
        print(f"Pruning adj to new {new_adj_test}")
        new_adj_test = new_adj_test.to(device)
        adj_attack, features_attack = attacker.attack(model=model,
                                                adj=new_adj_test,
                                                features=features,
                                                target_idx=target_idx,
                                                labels=attack_labels)
        n_total = features.size(0)
        new_adj_test = new_adj_test.cpu()
        new_x, new_y, _ = adj_attack[n_total:,:].coo()
        new_x += n_total
        x, y, _ = adj.coo()
        new_row = torch.cat((x,new_x,new_y),dim=0)
        new_col = torch.cat((y,new_y,new_x),dim=0)
        adj_attack = SparseTensor(row=new_row, col=new_col, value=torch.ones(new_row.size(0),device=device))
        print(f"Stick adj back to {adj_attack}")
    else:
        adj_attack, features_attack = attacker.attack(model=model,
                                                    adj=adj,
                                                    features=features,
                                                    target_idx=target_idx,
                                                    labels=attack_labels)
    return features_attack, adj_attack, target_idx


#run_GNN_frac_all.py --dataset Citeseer --cuda 0 --block constant_frac --function laplacian --time 5 --step_size 1 --hidden_dim 64 --lr 0.01 --input_dropout 0.4 --dropout 0.4 --runtime 1 --seed 123 --epoch 100 --decay 0.01 --method ceuler
def get_optimizer(name, parameters, lr, weight_decay=0):
  if name == 'sgd':
    return torch.optim.SGD(parameters, lr=lr, weight_decay=weight_decay)
  elif name == 'rmsprop':
    return torch.optim.RMSprop(parameters, lr=lr, weight_decay=weight_decay)
  elif name == 'adagrad':
    return torch.optim.Adagrad(parameters, lr=lr, weight_decay=weight_decay)
  elif name == 'adam':
    return torch.optim.Adam(parameters, lr=lr, weight_decay=weight_decay)
  elif name == 'adamax':
    return torch.optim.Adamax(parameters, lr=lr, weight_decay=weight_decay)
  else:
    raise Exception("Unsupported optimizer: {}".format(name))


def add_labels(feat, labels, idx, num_classes, device):
  onehot = torch.zeros([feat.shape[0], num_classes]).to(device)
  if idx.dtype == torch.bool:
    idx = torch.where(idx)[0]  # convert mask to linear index
  onehot[idx, labels.squeeze()[idx]] = 1

  return torch.cat([feat, onehot], dim=-1)


def get_label_masks(data, mask_rate=0.5):
  """
  when using labels as features need to split training nodes into training and prediction
  """
  if data.train_mask.dtype == torch.bool:
    idx = torch.where(data.train_mask)[0]
  else:
    idx = data.train_mask
  mask = torch.rand(idx.shape) < mask_rate
  train_label_idx = idx[mask]
  train_pred_idx = idx[~mask]
  return train_label_idx, train_pred_idx


def train(model, optimizer, data, train_idx):
  model.train()
  optimizer.zero_grad()
  feat = data.x ## cora:[2485,1433]
  if model.opt['use_labels']:
    train_label_idx, train_pred_idx = get_label_masks(data, model.opt['label_rate'])

    feat = add_labels(feat, data.y, train_label_idx, model.num_classes, model.device)
  else:
    train_pred_idx = data.train_mask

  #adv training 
  #step 1：生成对抗样本
  labels = data.y
  attack_labels = data.y if args.attack_label else None
  model.eval()
  if args.adv_train:
    if args.eval_attack.lower() == "pgd":
          attacker = PGD(epsilon=args.attack_lr, #决定了每次更新对抗样本时的最大扰动幅度。
                      n_epoch=args.adv_attack_epoch, # 控制 PGD 攻击的迭代次数。
                      n_inject_max= args.n_inject_max, #控制注入的对抗节点的最大数量。
                      n_edge_max= args.n_edge_max, #控制每个对抗节点可以连接的最大边数,每个对抗节点可以与图中的原始节点连接，形成新的边。
                      feat_lim_min=args.feat_lim_min, #控制对抗节点特征的最小值和最大值。
                      feat_lim_max=args.feat_lim_max,
                      device= model.device,
                      early_stop=args.early_stop) 
    ##adj_attack:sparsetensor形式，
     
    adj_attack, features_attack = attacker.adv_attack(model=model,
                                                    adj= data.edge_index,
                                                    features= feat,
                                                    target_idx=train_idx,
                                                    optimizer = optimizer,
                                                    labels=attack_labels)
  

  ##clean data
  out = model(feat, data.edge_index) #data.edge_index

  if model.opt['dataset'] == 'ogbn-arxiv':
    lf = torch.nn.functional.nll_loss
    loss = lf(out.log_softmax(dim=-1)[data.train_mask], data.y.squeeze(1)[data.train_mask])
  else:
    lf = torch.nn.CrossEntropyLoss()
    loss = lf(out[data.train_mask], data.y.squeeze()[data.train_mask])

  model.fm.update(model.getNFE())
  model.resetNFE()
  loss.backward()
  optimizer.step()
  model.bm.update(model.getNFE())
  model.resetNFE()

  return loss.item(), None#adv_loss.item()



@torch.no_grad()
def test_OGB(model, data, opt):


  feat = data.x
  if model.opt['use_labels']:
    feat = add_labels(feat, data.y, data.train_mask, model.num_classes, model.device)


  model.eval()

  if opt['dataset'] == 'ogbn-arxiv':
    name = 'ogbn-arxiv'
    evaluator = Evaluator(name=name)
    out = model(feat, data.edge_index).log_softmax(dim=-1)
    y_pred = out.argmax(dim=-1, keepdim=True)

    train_acc = evaluator.eval({
      'y_true': data.y[data.train_mask],
      'y_pred': y_pred[data.train_mask],
    })['acc']
    valid_acc = evaluator.eval({
      'y_true': data.y[data.val_mask],
      'y_pred': y_pred[data.val_mask],
    })['acc']
    test_acc = evaluator.eval({
      'y_true': data.y[data.test_mask],
      'y_pred': y_pred[data.test_mask],
    })['acc']


  return train_acc, valid_acc, test_acc

@torch.no_grad()
def test(model, data,  opt=None):  # opt required for runtime polymorphism
  model.eval()
  feat = data.x
  if model.opt['use_labels']:
    feat = add_labels(feat, data.y, data.train_mask, model.num_classes, model.device)
  logits, accs = model(feat, data.edge_index), []  ##
  logits = F.log_softmax(logits, dim=1)
  if opt['dataset'] in [ 'minesweeper', 'workers', 'questions']:
    # print("using ROC-AUC metric")
    for _, mask in data('train_mask', 'val_mask', 'test_mask'):
      # pred = logits.max(1)[1]
      # acc = pred.eq(data.y[mask]).sum().item() / mask.sum().item()
      mask_idx = torch.where(mask)[0]
      y_true = data.y[mask_idx].cpu().numpy()
      y_score = logits[mask_idx].cpu().numpy()
      acc = roc_auc_score(y_true=data.y[mask_idx].cpu().numpy(),
                                         y_score=logits[:, 1][mask_idx].cpu().numpy()).item()
      accs.append(acc)

  else:

    for _, mask in data('train_mask', 'val_mask', 'test_mask'):
      pred = logits[mask].max(1)[1]
      acc = pred.eq(data.y[mask]).sum().item() / mask.sum().item()
      accs.append(acc)
  return accs


def print_model_params(model):
  print(model)
  for name, param in model.named_parameters():
    if param.requires_grad:
      print(name)
      print(param.data.shape)


def merge_cmd_args(cmd_opt, opt):
  if cmd_opt['function'] is not None:
    opt['function'] = cmd_opt['function']
  if cmd_opt['block'] is not None:
    opt['block'] = cmd_opt['block']
  if cmd_opt['attention_type'] != 'scaled_dot':
    opt['attention_type'] = cmd_opt['attention_type']
  if cmd_opt['self_loop_weight'] is not None:
    opt['self_loop_weight'] = cmd_opt['self_loop_weight']
  if cmd_opt['method'] is not None:
    opt['method'] = cmd_opt['method']
  if cmd_opt['step_size'] != 1:
    opt['step_size'] = cmd_opt['step_size']
  if cmd_opt['time'] is not None:
    opt['time'] = cmd_opt['time']
  if cmd_opt['epoch'] is not None:
    opt['epoch'] = cmd_opt['epoch']
  if not cmd_opt['not_lcc']:
    opt['not_lcc'] = False
  if cmd_opt['num_splits'] != 1:
    opt['num_splits'] = cmd_opt['num_splits']
  if cmd_opt['dropout'] is not None:
    opt['dropout'] = cmd_opt['dropout']
  if cmd_opt['hidden_dim'] is not None:
    opt['hidden_dim'] = cmd_opt['hidden_dim']
  if cmd_opt['decay'] is not None:
    opt['decay'] = cmd_opt['decay']
  if cmd_opt['self_loop_weight'] is not None:
    opt['self_loop_weight'] = cmd_opt['self_loop_weight']
  if cmd_opt['edge_homo']  != 0:
    opt['edge_homo'] = cmd_opt['edge_homo']
  if cmd_opt['use_mlp'] is not None:
    opt['use_mlp'] = cmd_opt['use_mlp']
  if cmd_opt['data_norm'] is not None:
    opt['data_norm'] = cmd_opt['data_norm']

  if cmd_opt['lr'] is not None:
    opt['lr'] = cmd_opt['lr']
  if cmd_opt['input_dropout'] is not None:
    opt['input_dropout'] = cmd_opt['input_dropout']

  if cmd_opt['patience'] is not None:
    opt['patience'] = cmd_opt['patience']
  if cmd_opt['max_nfe'] is not None:
    opt['max_nfe'] = cmd_opt['max_nfe']


def set_seed(seed=123):
  random.seed(seed)
  np.random.seed(seed)
  os.environ['PYTHONHASHSEED'] = str(seed)
  torch.manual_seed(seed)
  torch.cuda.manual_seed(seed)
  torch.cuda.manual_seed_all(seed)
  torch.backends.cudnn.benchmark = False
  torch.backends.cudnn.deterministic = True

def get_optimizer_group(optimizer_name, grouped_parameters, **kwargs):
  if optimizer_name == 'adam':
    optimizer = torch.optim.Adam(grouped_parameters, **kwargs)
  elif optimizer_name == 'sgd':
    optimizer = torch.optim.SGD(grouped_parameters, **kwargs)
  elif optimizer_name == 'rmsprop':
    optimizer = torch.optim.RMSprop(grouped_parameters, **kwargs)
  elif optimizer_name == 'adagrad':
    optimizer = torch.optim.Adagrad(grouped_parameters, **kwargs)
  elif optimizer_name == 'adamax':
    optimizer = torch.optim.Adamax(grouped_parameters, **kwargs)
  # Add more optimizers here as needed
  else:
    raise ValueError("Invalid optimizer name")

  return optimizer

def combined_optimizer(model, opt):
  parameters_alphaode = [p for name, p in model.named_parameters() if p.requires_grad and 'alpha_ode' in name]
  parameters_other = [p for name, p in model.named_parameters() if p.requires_grad and 'alpha_ode' not in name]

  grouped_parameters = [
    {'params': parameters_other, 'lr': opt['lr'], 'weight_decay': opt['decay']},
    {'params': parameters_alphaode, 'lr': opt['lr'], 'weight_decay': opt['decay']}
  ]

  optimizer = get_optimizer_group(opt['optimizer'], grouped_parameters)
  return optimizer

def main(opt,split):

  args.best_weights = True
  args.inductive = True
  
   # adjust maximum injected nodes
  if args.grb_mode != 'full':  ##args.grb_mode:full  args.n_inject_max:60
        args.n_inject_max //= 3  
  
  set_seed(opt['seed'])
  dataset = get_dataset(opt, f'{ROOT_DIR}/data', opt['not_lcc'],split)
  # device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
  if opt['cuda'] >-1 :
    device = torch.device('cuda:' + str(opt['cuda']) if torch.cuda.is_available() else 'cpu')
  else:
    device = 'cpu'

  num_features = dataset.num_features #cora:1433 
  num_classes = dataset.num_classes #cora: 7 
  opt['num_classes'] = num_classes
  num_nodes = dataset.data.x.shape[0]  ##cora:2485
  opt['num_nodes'] = num_nodes

  print("num of nodes: ", num_nodes)
  print("num of features: ", num_features)
  print("num of classes: ", num_classes)

  model = GNN(opt, dataset, device).to(device)
  
  ##checkpoint加载
  checkpoint_path = '/amax/home/chenxiaotong/workspace/chy/FROND_remix3/pgd-stage_two/stage-first/model_2_87_40.pth'
  checkpoint = torch.load(checkpoint_path)
  missing_keys, unexpected_keys = model.load_state_dict(checkpoint, strict = False )
  ##
  model_dict = model.state_dict()
  # 额外打印被跳过的形状不匹配的键
  shape_mismatch_keys = [
        k for k in checkpoint.keys() 
        if k in model_dict and checkpoint[k].shape != model_dict[k].shape]
  print("成功加载的参数数量:", len(checkpoint) - len(shape_mismatch_keys))
  print("形状不匹配的键:", shape_mismatch_keys)
  print("缺失的键:", missing_keys)
  print("意外的键:", unexpected_keys)

  # 冻结成功加载的参数
  for name, param in model.named_parameters():

    if name in checkpoint and name not in shape_mismatch_keys :
        param.requires_grad = False
        print(f"冻结: {name}")
    else:
        print(f"未冻结: {name}")
    if name.startswith("cls_linear"):
        param.requires_grad = True
        print(f"强制可训练: {name}")
  # 验证冻结效果（可选）
  for name, param in model.named_parameters():
    print(f"参数: {name}, 可训练: {param.requires_grad}")
  
  #
  if not opt['planetoid_split'] and opt['dataset'] in ['Cora','Citeseer','Pubmed']:
    dataset.data = set_train_val_test_split(opt['seed'], dataset.data, num_development=5000 if opt["dataset"] == "CoauthorCS" else 1500)

  data = dataset.data.to(device)
  
  evaluator = Evaluator(name='ogbn-arxiv')
  transform = T.Compose([T.ToSparseTensor()])
  dataset_1 = Planetoid("/amax/home/chenxiaotong/workspace/chy/FROND_remix3/pgd/ICLR2024-FROND-main/data/Cora", args.dataset.lower(), transform=transform)
  data_1 = dataset_1[0]
  
  if args.dataset.lower() in ['cora','citeseer',"computers"] or args.dataset.lower().startswith("grb-"):
     split_idx = { 'train': torch.nonzero(data.train_mask, as_tuple=True)[0],
                  'valid':torch.nonzero(data.val_mask, as_tuple=True)[0], 
                  'test': torch.nonzero(data.test_mask, as_tuple=True)[0]}

  train_idx = split_idx['train'].to(device)##shape:140
  val_idx = split_idx['valid'].to(device) #shape:1360
  test_idx = split_idx['test'].to(device) #shape: 985

  if args.inductive: ##here
        # inductive split will automatically use relative ids for splitted graphs
        adj_train, adj_val, adj_test = inductive_split(data.edge_index, split_idx)#data_1.adj_t  #data.edge_index
        x_train, y_train = data.x[train_idx], data.y[train_idx]
        train_val_idx, _ = torch.sort(torch.cat([train_idx,val_idx],dim=0))
        x_val, y_val = data.x[train_val_idx], data.y[val_idx]
        x_test, y_test = data.x, data.y[test_idx]
        # tval_idx_train, tval_idx_val = get_index_induc(train_idx,val_idx)
        # tval_idx_train = torch.LongTensor(tval_idx_train).to(device)
        # tval_idx_val = torch.LongTensor(tval_idx_val).to(device)
  else:
        adj_train =  adj_val =  adj_test = data.adj_t
        x_train = x_val = x_test = data.x
        y_train = y_val = y_test = data.y
        
        
  robo_tests = []
  batch_robo_tests = {}


  data.edge_index = to_undirected(data.edge_index)
  print("num of train samples: ", len(torch.nonzero(data.train_mask,as_tuple=True)[0]))
  print("num of val samples: ", len(torch.nonzero(data.val_mask,as_tuple=True)[0]))
  print("num of test samples: ", len(torch.nonzero(data.test_mask,as_tuple=True)[0]))

  parameters = [p for p in model.parameters() if p.requires_grad]
  print_model_params(model)
  optimizer = combined_optimizer(model, opt)


  best_time = best_epoch = train_acc = val_acc = test_acc = 0

  this_test = test_OGB if opt['dataset'] == 'ogbn-arxiv' else test
  counter = 0
  
  #adv train metric
  acc_trace_train_pert = []
  robust_epsilon = 0.1

  for epoch in range(1, opt['epoch']):
    start_time = time.time()


    ##clean train
    loss, adv_loss = train(model, optimizer, data, train_idx)
    tmp_train_acc, tmp_val_acc, tmp_test_acc = this_test(model, data, opt)

    best_time = opt['time']
    if tmp_val_acc > val_acc:
      best_epoch = epoch
      train_acc = tmp_train_acc
      val_acc = tmp_val_acc
      test_acc = tmp_test_acc
      best_time = opt['time']
      counter = 0
      
      if args.best_weights:
        best_weights = deepcopy(model.state_dict())
      
      
    else:
      counter = counter + 1
      if counter == opt['patience']:
        break

    log = 'Epoch: {:03d}, Runtime {:03f}, Loss {:03f}, forward nfe {:d}, backward nfe {:d}, Train: {:.4f}, Val: {:.4f}, Test: {:.4f}, Best time: {:.4f}'

    print(log.format(epoch, time.time() - start_time, loss, model.fm.sum, model.bm.sum, tmp_train_acc, tmp_val_acc, tmp_test_acc, best_time))
  print('best val accuracy {:03f} with test accuracy {:03f} at epoch {:d} and best time {:03f}'.format(val_acc, test_acc,
                                                                                                     best_epoch,
                                                                                                     best_time))
  # #  ##保存模型
  # save_name = f"model_{split}_{best_epoch}_{best_time}.pth"
  # save_path = os.path.join(args.out_path, save_name)
  # torch.save(model.state_dict(),save_path)
  # print(f'模型已保存:{save_path}')

  run = 0
  if args.eval_robo and not args.batch_eval:
            if args.best_weights and args.epoch>0:
                model.load_state_dict(best_weights)
            test_idx = split_idx["test"].to(device)
            
            target_idx = test_idx
            x_attack, adj_attack, target_idx = eval_robustness(model, x_test, adj_test, target_idx, data.y, device, args, run)
            
            x_new = torch.cat([x_test,x_attack],dim=0) if x_attack != None else x_test
            if len(args.save_attack) > 0 and not args.eval_robo_blk:
                atkg_path = os.path.join(args.save_attack,args.dataset)+f"_{args.eval_attack}"
                if not os.path.exists(atkg_path):
                    os.makedirs(atkg_path)
                # targeted attack
                if args.eval_target:
                    atkg_path += "_target"
                # multi-split eval
                if args.mul_run>0 and run > 0:
                    atkg_path +=f"_{run}.pt"
                else:
                    atkg_path += ".pt"

                print(f"saving the generated atkg to {atkg_path}")
                # saving format of the perturbed graph
                adj_row, adj_col = adj_attack.coo()[:2]
                new_data = Data(edge_index=torch.stack([adj_row,adj_col], dim=0),
                                x=x_new,y=data.y)
                new_data.train_mask = data.train_mask
                new_data.val_mask = data.val_mask
                new_data.test_mask= data.test_mask
                new_data.target_idx= target_idx
                # new_data.orig_edge_size = adj_test.coo()[0].size(0)
                torch.save(new_data.cpu(),atkg_path)

            tst = sep_test(model,x_new,adj_attack,data.y,target_idx,evaluator)
            
            log = 'Test robustness accuracy: {:03f}'
            print(log.format(tst))
             
            print(f"Test robustness accuracy: {tst}")
  elif args.batch_eval:
            if args.best_weights and args.epochs>0:
                model.load_state_dict(best_weights)
            target_idx = test_idx
            for (i,atk) in enumerate(args.batch_attacks):
                for j in range(max(args.mul_run,1)):
                    # not necessary to test vanilla, rnd, speit multiple times
                    if j>=1 and atk.lower() in ["vanilla","rnd","speitml"]:
                        continue
                    args.eval_attack = atk
                    x_attack, adj_attack, target_idx = eval_robustness(model, x_test, adj_test, target_idx, data.y, device, args, run=j)
                    
                    x_new = torch.cat([x_test, x_attack], dim=0) if x_attack != None else x_test
                    tst = sep_test(model, x_new, adj_attack, data.y, target_idx, evaluator)
                    if run == 0:
                        batch_robo_tests[atk] = [tst]
                    else:
                        batch_robo_tests[atk].append(tst)
                    print(f"Test robustness accuracy under {atk}: {tst}")
                    # save gpu memory
                    x_attack.cpu()
                    adj_attack.cpu()
                    target_idx.cpu()
                    torch.cuda.empty_cache()
  
  
  return train_acc, val_acc, test_acc, opt, tst


if __name__ == '__main__':

  args = parser.parse_args()

  cmd_opt = vars(args)

  try:
    best_opt = best_params_dict[cmd_opt['dataset']]
    opt = {**cmd_opt, **best_opt}
    merge_cmd_args(cmd_opt, opt)
  except KeyError:
    opt = cmd_opt

  best = []
  robust_best = []
  timestr = time.strftime("%H%M%S")

  # mkdir for log
  if not os.path.exists("/amax/home/chenxiaotong/workspace/chy/FROND_remix3/pgd-two-stage-adv-training/ICLR2024-FROND-main/log_frac"):
    os.makedirs("log_frac")
  filename = "/amax/home/chenxiaotong/workspace/chy/FROND_remix3/pgd-two-stage-adv-training/ICLR2024-FROND-main/log_frac/" + str(args.dataset) + str(args.method) + str(args.function) + str(args.block) + str(
    args.time) + timestr + ".txt"
  command_args = " ".join(sys.argv)
  with open(filename, 'a') as f:
    json.dump(command_args, f)
    f.write("\n")

  for i in range(opt['runtime']):
    opt['seed'] = opt['seed'] + i
    train_acc, val_acc, test_acc, opt_final,robo_tests = main(opt,i)

    best.append(test_acc)
    with open(filename, 'a') as f:
      json.dump(test_acc, f)
      f.write("\n")
    print("test acc: ", best)
    # opt['seed'] += 1
    
    
     ##robustness
    robust_best.append(robo_tests)
    with open(filename, 'a') as f:
      f.write("robustness:")
      json.dump(robo_tests, f)
      f.write("\n")
    print("robust_test acc: ", robust_best)
    # opt['seed'] += 1
    
    
  print('Mean test accuracy: ', np.mean(np.array(best) * 100), 'std: ', np.std(np.array(best) * 100))
  print("test acc: ", best)
  
 
  print('Mean robust_test accuracy: ', np.mean(np.array(robust_best) * 100), 'std: ', np.std(np.array(robust_best) * 100))
  print("robust_test acc: ", robust_best)
  
  

  with open(filename, 'a') as f:
    f.write(str(np.mean(np.array(best) * 100)))
    f.write(",")
    f.write(str(np.std(np.array(best) * 100)))
    f.write("\n")
    json.dump(opt_final, f, indent=2)
  # change file name to include best test acc
  os.rename(filename, filename[:-4] + str(np.mean(np.array(best) * 100)) + ".txt")





