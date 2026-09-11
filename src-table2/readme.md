## Requirements

To install the required dependencies, refer to the environment.yaml file

## Reproducing Results


To reproduce the adversarial defense results against graph injection attacks （GIA） shown in Table 1 in the manuscript, run the following command: 


```bash
#Cora dataset
#pgd
python -u run_GNN_frac_all-BaseGNN+laya-stage2-newsplit.py --dataset 'cora'  --inductive --eval_robo --eval_attack 'pgd' --n_inject_max 60 --n_edge_max 20 --grb_mode 'full' --runs 1 --disguise_coe 0 --use_ln 0 --injection 'random' --grb_split  --input_dropout 0.8  --dropout 0.4 --lr 0.05  --decay 0.01 --epoch 200  --time 2  --checkpoint_path ckpt-pgd/model_3_167_3.pth --runtime 10 --function transformer

#tdgia
python -u run_GNN_frac_all-BaseGNN+laya-stage2-newsplit.py --dataset 'cora'  --inductive --eval_robo --eval_attack 'tdgia' --n_inject_max 60 --n_edge_max 20 --grb_mode 'full' --runs 1 --disguise_coe 0 --use_ln 0 --injection 'random' --grb_split  --input_dropout 0.8  --dropout 0.4 --lr 0.05  --decay 0.01 --epoch 200  --time 2  --checkpoint_path ckpt-tdgia/model_3_167_3.pth -runtime 10 --function transformer

#metagia
python -u run_GNN_frac_all-BaseGNN+laya-stage2-newsplit.py --dataset 'cora'  --inductive --eval_robo --eval_attack 'metagia' --n_inject_max 60 --n_edge_max 20 --grb_mode 'full' --runs 1 --disguise_coe 0 --use_ln 0 --injection 'random' --grb_split  --input_dropout 0.8  --dropout 0.4 --lr 0.01  --decay 0.01 --epoch 800  --time 40  --checkpoint_path ckpt-metagia/model_9_230_40.pth -runtime 10 --function transformer 

```
