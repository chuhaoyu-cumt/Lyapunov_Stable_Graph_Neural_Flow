
## Requirements

To install the required dependencies, refer to the environment.yaml file

## Reproducing Results


For the non-targeted GIA in Table 1, first generate the adversarial graphs , for Cora dataset run the following command:

```bash
#pgd
python -u gnn_misg.py --dataset 'cora'  --inductive --eval_robo --eval_attack 'pgd' --n_inject_max 60 --n_edge_max 20 --grb_mode 'full' --runs 1 --disguise_coe 0  --use_ln 0 --grb_split
#tdgia
python -u gnn_misg.py --dataset 'cora'  --inductive --eval_robo --eval_attack 'seqgia' --n_inject_max 60 --n_edge_max 20 --grb_mode 'full' --runs 1 --disguise_coe 0 --use_ln 0 --injection 'tdgia' --grb_split
cp atkg/cora_seqgia.pt atkg/cora_tdgia.pt

#metagia
python -u gnn_misg.py --dataset 'cora'  --inductive --eval_robo --eval_attack 'seqgia' --injection 'meta' --n_inject_max 60 --n_edge_max 20 --grb_mode 'full' --runs 1 --disguise_coe 0 --use_ln 0  --grb_split
cp atkg/cora_seqgia.pt atkg/cora_metagia.pt

```
