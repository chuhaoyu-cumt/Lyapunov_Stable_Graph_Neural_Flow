from base_classes import ODEblock
import torch
from utils import get_rw_adj, gcn_norm_fill_val
from torchfde import fdeint
from stablemodels.lyapunov import LyapunovFunction
from torch import nn

import torch_sparse
from torch_sparse import SparseTensor
from torch_geometric.utils import to_edge_index
class ConstantODEblock_FRAC(ODEblock):
  def __init__(self, odefunc,  opt, data,  device, t=torch.tensor([0, 1])):
    super(ConstantODEblock_FRAC, self).__init__(odefunc,  opt,  data, device, t)

    self.odefunc = odefunc(opt['hidden_dim'], opt['hidden_dim'], opt, data, device)
   

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
    lyapunov_lr = 2e-4
    lyapunov_eps = 1e-3
    self._alpha = 0.9
    self._obs_size = opt['hidden_dim']
    layer_sizes = [ opt['hidden_dim'], opt['hidden_dim']]

    self._l1 = nn.Linear(input_shape + control_size, opt['hidden_dim'])
    self._l2 = nn.Linear(opt['hidden_dim'], opt['hidden_dim'])
    self._l3 = nn.Linear(opt['hidden_dim'], input_shape)
    
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
       
      edge_index, edge_weight = gcn_norm_fill_val(self.edge_index, edge_weight=self.edge_attr,
                                           fill_value=self.opt['self_loop_weight'],
                                           num_nodes=int(x.shape[0]),
                                           dtype=x.dtype)
    else:
      self.edge_index = adj
      self.edge_attr = None
        
      edge_index, edge_weight = gcn_norm_fill_val(self.edge_index, edge_weight=self.edge_attr,
                                           fill_value=self.opt['self_loop_weight'],
                                           num_nodes=int(x.shape[0]),
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

    
    z = fdeint(func, state, alpha, t=self.opt['time'], step_size=self.opt['step_size'], method=self.opt['method']) ##对动力系统进行求解
    


    if not z.requires_grad:
            z.requires_grad = True

    lyapunov = self._lyapunov_function(z)
   

    z.retain_grad()
    lyapunov.backward(gradient=torch.ones_like(lyapunov), retain_graph=True)
    grad_v = z.grad.clone() 
    #print(f"grad_v:{grad_v.shape}")
    gv = grad_v.view(-1, 1, *z.shape[1:])
    fv = grad_v.view(-1, *z.shape[1:], 1)
    dot = (gv @ fv).squeeze() 
    #print(f"dot:{dot.shape}")
    orth = (dot + self._alpha * lyapunov).squeeze().relu() / grad_v.pow(2).sum() 
    if len(orth.shape) == 0:
            orth.unsqueeze_(0)
   

    return z - (orth.T * grad_v)[:, :self._obs_size]

  def __repr__(self):
    return self.__class__.__name__ + '( Time Interval ' + str(self.t[0].item()) + ' -> ' + str(self.t[1].item()) \
           + ")"
