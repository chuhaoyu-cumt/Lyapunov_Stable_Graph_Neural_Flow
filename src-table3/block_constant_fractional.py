from base_classes import ODEblock
import torch
from utils import get_rw_adj, gcn_norm_fill_val
from torchfde import fdeint
from stablemodels.lyapunov import LyapunovFunction
from torch import nn

import torch_sparse
from torch_sparse import SparseTensor

class ConstantODEblock_FRAC(ODEblock):
  def __init__(self, odefunc,  opt, data,  device, t=torch.tensor([0, 1])):
    super(ConstantODEblock_FRAC, self).__init__(odefunc,  opt,   device, t)

    self.odefunc = odefunc(opt['hidden_dim'], opt['hidden_dim'], opt, device)
    # if opt['data_norm'] == 'rw':
    #   edge_index, edge_weight = get_rw_adj(data.edge_index, edge_weight=data.edge_attr, norm_dim=1,
    #                                                                fill_value=opt['self_loop_weight'],
    #                                                                num_nodes=data.num_nodes,
    #                                                                dtype=data.x.dtype)
    # else:
    #   edge_index, edge_weight = gcn_norm_fill_val(data.edge_index, edge_weight=data.edge_attr,
    #                                        fill_value=opt['self_loop_weight'],
    #                                        num_nodes=data.num_nodes,
    #                                        dtype=data.x.dtype)
    # self.odefunc.edge_index = edge_index.to(device)
    # self.odefunc.edge_weight = edge_weight.to(device)
    # self.reg_odefunc = None
    # self.reg_odefunc.odefunc.edge_index, self.reg_odefunc.odefunc.edge_weight = self.odefunc.edge_index, self.odefunc.edge_weight

    if opt['adjoint']:
      from torchdiffeq import odeint_adjoint as odeint
    else:
      from torchdiffeq import odeint

    self.train_integrator = odeint
    self.test_integrator = odeint
    # self.set_tol()
    self.device = device
    self.opt = opt
    
    input_shape =  opt['hidden_dim']
    control_size = 0
    lyapunov_lr = 3e-4
    lyapunov_eps = 1e-3
    layer_sizes = [ opt['hidden_dim'], opt['hidden_dim']]
    
    self._lyapunov_function = LyapunovFunction((input_shape + control_size,),
                                                   layer_sizes=layer_sizes ,
                                                   lr=lyapunov_lr,
                                                   eps=lyapunov_eps).to(device)
    
    self.proj_lyapunov = nn.Linear( opt['hidden_dim'], opt['hidden_dim']).to(device)
    
    
    
  def forward(self, x,adj):
    
    if type(adj) is torch_sparse.tensor.SparseTensor:
      row, col, _ = adj.coo()
      self.edge_index = torch.stack([row.to(self.device), col.to(self.device)], dim=0).to(self.device)
      self.edge_attr = None
       
      edge_index, edge_weight = get_rw_adj(self.edge_index, edge_weight=self.edge_attr, norm_dim=1,
                                           fill_value=self.opt['self_loop_weight'],
                                           num_nodes=self.opt['num_nodes'] + self.opt['n_inject_max'],
                                           dtype=x.dtype)
    else:
      self.edge_index = adj
      self.edge_attr = None
        
      edge_index, edge_weight = get_rw_adj(self.edge_index, edge_weight=self.edge_attr, norm_dim=1,
                                              fill_value=self.opt['self_loop_weight'],
                                              num_nodes=self.opt['num_nodes'],
                                              dtype=x.dtype)
      
      
    self.odefunc.edge_index = edge_index.to(self.device)
    self.odefunc.edge_weight = edge_weight.to(self.device)
    
    
    
    
    t = self.t.type_as(x)

    integrator = self.train_integrator if self.training else self.test_integrator
    
    # reg_states = tuple( torch.zeros(x.size(0)).to(x) for i in range(self.nreg) )

    # func = self.reg_odefunc if self.training and self.nreg > 0 else self.odefunc
    # state = (x,) + reg_states if self.training and self.nreg > 0 else x

    func = self.odefunc
    state = x


    alpha = torch.tensor(self.opt['alpha_ode'])

    if alpha > 1:
        raise ValueError("alpha_ode must be in (0,1)")
    
    ##求解器 func=投影之后李雅普诺夫稳定的动力系统  z=输出，去做分类
    z = fdeint(func, state, alpha, t=self.opt['time'], step_size=self.opt['step_size'], method=self.opt['method']) ##对动力系统进行求解
    
    # print(z.shape)
    
    # ###
    # ##定义一个投影层，linear
    f_proj = self.proj_lyapunov(z)
      
     ##计算liyapunuofu空间 特征
    #针对每个样本都要映射遍历样本
      
    f_lyapunov_list = []
      
    for i in range(f_proj.shape[0]):
        f_lyapunov_i = self._lyapunov_function(f_proj[i].unsqueeze(0))
        f_lyapunov_list.append(f_lyapunov_i.T)
    
        
    f_lyapunov = torch.stack(f_lyapunov_list, dim = 0)

    
    

    return f_lyapunov.squeeze(1)#z ##z输入图网络分类
    


  def __repr__(self):
    return self.__class__.__name__ + '( Time Interval ' + str(self.t[0].item()) + ' -> ' + str(self.t[1].item()) \
           + ")"
