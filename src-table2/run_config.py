import argparse
import os
parser = argparse.ArgumentParser()
parser.add_argument('--use_cora_defaults', action='store_true',
                  help='Whether to run with best params for cora. Overrides the choice of dataset')
parser.add_argument('--cuda', default=0, type=int) ##指定GPU？
# data args
parser.add_argument('--dataset', type=str, default='Cora',
                  help='Cora, Citeseer, Pubmed, Computers, Photo, CoauthorCS, ogbn-arxiv,chameleon, squirrel,'
                       'wiki-cooc, roman-empire, amazon-ratings, minesweeper, workers, questions',)
parser.add_argument('--data_norm', type=str, default='gcn',
                  help='rw for random walk, gcn for symmetric gcn norm')
parser.add_argument('--self_loop_weight', default=1,type=float, help='Weight of self-loops.')
parser.add_argument('--use_labels', dest='use_labels', action='store_true', help='Also diffuse labels')
parser.add_argument('--geom_gcn_splits', default=True, dest='geom_gcn_splits', action='store_true',
                  help='use the 10 fixed splits from '
                       'https://arxiv.org/abs/2002.05287')
parser.add_argument('--num_splits', type=int, dest='num_splits', default=1,
                  help='the number of splits to repeat the results on')
parser.add_argument('--label_rate', type=float, default=0.5,
                  help='% of training labels to use when --use_labels is set.')
parser.add_argument('--planetoid_split', action='store_true',
                  help='use planetoid splits for Cora/Citeseer/Pubmed')

parser.add_argument('--random_splits',default = 'False', action='store_true',help='fixed_splits')

parser.add_argument('--edge_homo', type=float, default=0.0, help="edge_homo")
parser.add_argument('--out_path', type=str, default='/AAAI26-update/stage-2/src/stage_two_baseGNN-ori+laya/')

parser.add_argument('--checkpoint_path', type=str, default='/workspace/chy/AAAI26-update/stage-2/src/ckpt/model_3_167_3.pth')
# GNN args
parser.add_argument('--hidden_dim',default=64, type=int,  help='Hidden dimension.')
parser.add_argument('--fc_out', dest='fc_out', action='store_true',
                  help='Add a fully connected layer to the decoder.')
parser.add_argument('--input_dropout', type=float,default=0.8,  help='Input dropout rate.')
parser.add_argument('--dropout', type=float,default=0.4, help='Dropout rate.')#0.4->0.8
parser.add_argument("--batch_norm", dest='batch_norm', default = True, help='search over reg params')
parser.add_argument('--optimizer', type=str, default='adam', help='One from sgd, rmsprop, adam, adagrad, adamax.')
parser.add_argument('--lr', type=float, default=0.05, help='Learning rate.')
parser.add_argument('--decay', type=float,default=0.01,  help='Weight decay for optimization')
parser.add_argument('--epoch', type=int, default=200, help='Number of training epochs per iteration.')#200
parser.add_argument('--alpha', type=float, default=1.0, help='Factor in front matrix A.')
parser.add_argument('--alpha_dim', type=str, default='sc', help='choose either scalar (sc) or vector (vc) alpha')
parser.add_argument('--no_alpha_sigmoid', dest='no_alpha_sigmoid', action='store_true',
                  help='apply sigmoid before multiplying by alpha')
parser.add_argument('--beta_dim', type=str, default='sc', help='choose either scalar (sc) or vector (vc) beta')
parser.add_argument('--block', default='constant_frac',type=str,  help='constant_graph,constant, mixed, attention, hard_attention')#constant_frac:ConstantODEblock_FRAC   att_frac:AttODEblock_FRAC 
parser.add_argument('--function',default='transformer', type=str, help='transgrand,belgrand,laplacian, transformer, dorsey, GAT') #transgraphcon:ODEFuncAtt_graphcon   transformer:ODEFuncTransformerAtt
parser.add_argument('--use_mlp', type=bool,
                  help='Add a fully connected layer to the encoder.')
parser.add_argument('--add_source', dest='add_source', default = True,
                  help='If try get rid of alpha param and the beta*x0 source term')
parser.add_argument('--cgnn', dest='cgnn', action='store_true', help='Run the baseline CGNN model from ICML20')

parser.add_argument('--patience', type=int, default=100, help='Number of training patience per iteration.')

# ODE args
parser.add_argument('--time',default=2,  type=float, help='End time of ODE integrator.')#  
parser.add_argument('--augment', action='store_true',
                  help='double the length of the feature vector by appending zeros to stabilist ODE learning')
parser.add_argument('--method',default='predictor',  type=str, help="set the numerical solver: dopri5, euler, rk4, midpoint")
parser.add_argument('--step_size', type=float, default=1,
                  help='fixed step size when using fixed step solvers e.g. rk4')
parser.add_argument('--max_iters', type=float, default=100, help='maximum number of integration steps')
parser.add_argument("--adjoint_method", type=str, default="adaptive_heun",
                  help="set the numerical solver for the backward pass: dopri5, euler, rk4, midpoint")
parser.add_argument('--adjoint', dest='adjoint', action='store_true',
                  help='use the adjoint ODE method to reduce memory footprint')
parser.add_argument('--adjoint_step_size', type=float, default=1,
                  help='fixed step size when using fixed step adjoint solvers e.g. rk4')
parser.add_argument('--tol_scale', type=float, default=1., help='multiplier for atol and rtol')
parser.add_argument("--tol_scale_adjoint", type=float, default=1.0,
                  help="multiplier for adjoint_atol and adjoint_rtol")
parser.add_argument('--ode_blocks', type=int, default=1, help='number of ode blocks to run')
parser.add_argument("--max_nfe", type=int, default=100000000000,
                  help="Maximum number of function evaluations in an epoch. Stiff ODEs will hang if not set.")
parser.add_argument("--no_early", action="store_true",
                  help="Whether or not to use early stopping of the ODE integrator when testing.")
parser.add_argument('--earlystopxT', type=float, default=3, help='multiplier for T used to evaluate best model')
parser.add_argument("--max_test_steps", type=int, default=100,
                  help="Maximum number steps for the dopri5Early test integrator. "
                       "used if getting OOM errors at test time")

# Attention args
parser.add_argument('--leaky_relu_slope', type=float, default=0.2,
                  help='slope of the negative part of the leaky relu used in attention')
parser.add_argument('--attention_dropout', type=float, default=0., help='dropout of attention weights')
parser.add_argument('--heads', type=int, default=2, help='number of attention heads')#ori:4
parser.add_argument('--attention_norm_idx', type=int, default=0, help='0 = normalise rows, 1 = normalise cols')
parser.add_argument('--attention_dim', type=int, default=64,
                  help='the size to project x to before calculating att scores')
parser.add_argument('--mix_features', dest='mix_features', action='store_true',
                  help='apply a feature transformation xW to the ODE')
parser.add_argument('--reweight_attention', dest='reweight_attention', action='store_true',
                  help="multiply attention scores by edge weights before softmax")
parser.add_argument('--attention_type', type=str, default="scaled_dot",
                  help="scaled_dot,cosine_sim,pearson, exp_kernel")
parser.add_argument('--square_plus', action='store_true', help='replace softmax with square plus')



# rewiring args
parser.add_argument("--not_lcc", action="store_false", help="don't use the largest connected component")


parser.add_argument('--alpha_ode', type=float, default=0.85, help='alpha_ode')#0.85->0.55
parser.add_argument('--runtime', type=int, default=10, help="runtime")
parser.add_argument('--seed', type=int, default=123, help="seed")



##robustness
parser.add_argument('--batch_eval', action="store_true") 
parser.add_argument('--batch_attacks', type=list, default=[])

 # enforce multi-run evaluation
parser.add_argument('--mul_run', type=int, default=0)
######################## Robustness Eval Setting ####################
parser.add_argument('--eval_robo', default = True )#action="store_true"
    # targeted attack else non-targeted
parser.add_argument('--eval_target', action="store_true")
    # number of targets in each deg category
parser.add_argument('--target_num', type=int, default=200) 
    
# if evaluated in blackbox, the attacked graph will be loaded for evaluation
parser.add_argument('--eval_robo_blk', default = True)#action="store_true"
# the attack method used for evaluation
parser.add_argument('--eval_attack', type=str, default="tdgia")
    # maximum number of injected nodes at 'full' data mode
    # if in other data modes, e.g., 'easy', it shall be 1/3 of that in 'full' mode
parser.add_argument('--n_inject_max', type=int, default=60)
    # maximum number of edges of the injected (per) node 
parser.add_argument('--n_edge_max', type=int, default=20)
    # attack feat limit, if not spec_feat_lim, auto calculate from data.x
parser.add_argument('--spec_feat_lim', action="store_true")
parser.add_argument('--feat_lim_min', type=float, default=-1.0)
parser.add_argument('--feat_lim_max', type=float, default=1.0)
    # attack feature update epochs
parser.add_argument('--attack_epoch', type=int, default=500)#500
    # attack A_atk update epochs
parser.add_argument('--agia_epoch', type=int, default=300)
    # how much vicious nodes being injected randomly before agia is applied
parser.add_argument('--agia_pre', type=float, default=0.5)
    # number of iterative epochs for agia
parser.add_argument('--iter_epoch', type=int, default=200)
    # attack step size
parser.add_argument('--attack_lr', type=float, default=0.01)
    # early stopping feat upd for attack
parser.add_argument('--early_stop', type=int, default=2)
    # weight of the disguised regularization term
parser.add_argument('--disguise_coe', type=float, default=0)
parser.add_argument('--hinge', action="store_true")
    # update features with label information if set true
parser.add_argument('--attack_label', action="store_true")
    # save path of the attacked feature and graph
parser.add_argument('--save_attack', type=str, default="/workspace/chy/AAAI26-update/atkg/")
    
    # use corresponding subgraph for attack
parser.add_argument('--prune_graph', action="store_true")

    # paramters for seqgia
parser.add_argument('--sequential_step', type=float, default=0.2)
parser.add_argument('--injection', type=str, default="random") 
parser.add_argument('--feat_upd', type=str, default="gia")
parser.add_argument('--branching', action="store_true")
parser.add_argument('--grb_mode',type=str,default='full')
 # enforce grb split
parser.add_argument('--grb_split',default=True )#action="store_true"
parser.add_argument('--inductive', default=True )#action="store_true"

parser.add_argument('--runs', type=int, default=1)
# put layer norm between layers or not
parser.add_argument('--use_ln', type=int,default=0)


##adv_train
parser.add_argument('--adv_train', default = 'true')
parser.add_argument('--adv_attack_epoch', type = int, default =100)


parser.add_argument('--no_alpha', action="store_true")
