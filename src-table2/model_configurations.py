from function_transformer_attention import ODEFuncTransformerAtt



from block_constant_fractional import ConstantODEblock_FRAC


from function_beltrami_trans import ODEFuncBeltramiAtt
from function_transformer_grand import ODEFuncTransformerAtt_GRAND


class BlockNotDefined(Exception):
  pass

class FunctionNotDefined(Exception):
  pass


def set_block(opt):
  ode_str = opt['block']
  if ode_str == 'constant_frac':
    block = ConstantODEblock_FRAC
  else:
    raise BlockNotDefined
  return block


def set_function(opt):
  ode_str = opt['function']
  if ode_str == 'transformer':
    f = ODEFuncTransformerAtt
  elif ode_str == 'belgrand':
    f = ODEFuncBeltramiAtt
  elif ode_str == 'transgrand':
    f = ODEFuncTransformerAtt_GRAND
  else:
    raise FunctionNotDefined
  return f
