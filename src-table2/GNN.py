import torch
from torch import nn
import torch.nn.functional as F
from base_classes import BaseGNN
from model_configurations import set_block, set_function
from torch.nn.parameter import Parameter
import math
import geotorch


class newLinear(nn.Module):
    def __init__(self, in_features, out_features, bias=True):
        super(newLinear, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.weight = Parameter(torch.Tensor(in_features,out_features))
#         self.weight = self.weighttemp.T
        if bias:
            self.bias = Parameter(torch.Tensor(out_features))
        else:
            self.register_parameter('bias', None)
        self.reset_parameters()

    def reset_parameters(self):
        nn.init.kaiming_uniform_(self.weight, a=math.sqrt(5))
        if self.bias is not None:
            fan_in, _ = nn.init._calculate_fan_in_and_fan_out(self.weight)
            bound = 1 / math.sqrt(fan_in)
            nn.init.uniform_(self.bias, -bound, bound)

    def forward(self, input):
        return F.linear(input, self.weight.T, self.bias)

    def extra_repr(self) -> str:
        return 'in_features={}, out_features={}, bias={}'.format(
            self.in_features, self.out_features, self.bias is not None
        )

class ORTHFC(nn.Module):
    def __init__(self, dimin, dimout, bias):
        super(ORTHFC, self).__init__()
        if dimin >= dimout:
            self.linear = newLinear(dimin, dimout,  bias=bias)
        else:
            self.linear = nn.Linear(dimin, dimout,  bias=bias)
        geotorch.orthogonal(self.linear, "weight")

    def forward(self, x):
        return self.linear(x)
    
class MLP_OUT_ORTH(nn.Module):
    def __init__(self,opt,num_classes):
        super(MLP_OUT_ORTH, self).__init__()
        self.fc0 = ORTHFC(opt['hidden_dim'], num_classes, False)  ##self.fc0 = ORTHFC(64, 10 False)
    def forward(self, input_):
        h1 = self.fc0(input_)
        return h1

# Define the GNN model.
class GNN(BaseGNN):
  def __init__(self, opt, dataset, device=torch.device('cpu')):
    super(GNN, self).__init__(opt, dataset, device)
    self.f = set_function(opt) ##动力系统？
    block = set_block(opt) ##求解器
    time_tensor = torch.tensor([0, self.T]).to(device)
    self.odeblock = block(self.f, opt, dataset.data, device, t=time_tensor).to(device) #
    # self.alpha_ode = nn.Parameter(torch.tensor(torch.tensor(0.1), requires_grad=True))
    # print("self.alpha_ode: ",self.alpha_ode )
    # self.linear_term = nn.Linear(opt['hidden_dim'] * opt['num_terms'], opt['hidden_dim'])
    
    self.cls_linear = MLP_OUT_ORTH(opt,self.num_classes).to(device)

  def forward(self, x, adj):
    # Encode each node based on its feature.
    if self.opt['use_labels']:
      y = x[:, -self.num_classes:]
      x = x[:, :-self.num_classes]


    x = F.dropout(x, self.opt['input_dropout'], training=self.training)
    x = self.m1(x) ##linear函数  output:[2485,64]

    if self.opt['use_mlp']: #设置为0
      x = F.dropout(x, self.opt['dropout'], training=self.training)
      x = F.dropout(x + self.m11(F.relu(x)), self.opt['dropout'], training=self.training)
      x = F.dropout(x + self.m12(F.relu(x)), self.opt['dropout'], training=self.training)
    # todo investigate if some input non-linearity solves the problem with smooth deformations identified in the ANODE paper

    if self.opt['use_labels']:
      x = torch.cat([x, y], dim=-1)

    if self.opt['batch_norm']:
      x = self.bn_in(x)

    # Solve the initial value problem of the ODE.
    x_init = x.clone()

    if 'graphcon' in self.opt['function']:
      x = torch.cat([x, x_init], dim=-1)
      self.odeblock.set_x0(x)
      z = self.odeblock(x,adj)
      z = z[:,self.opt['hidden_dim']:]
    else:
      self.odeblock.set_x0(x) ##不反向传播
      z = self.odeblock(x,adj)



    # Activation.
    z = F.relu(z)

    if self.opt['fc_out']:
      z = self.fc(z)
      z = F.relu(z)

    # Dropout.
    z = F.dropout(z, self.opt['dropout'], training=self.training)
    
    z1=z
    
    z_new = self.cls_linear(z1)
  
    
    return z_new

