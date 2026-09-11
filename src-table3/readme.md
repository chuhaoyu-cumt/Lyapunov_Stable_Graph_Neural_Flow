To reproduce the adversarial defense results against graph meta attacks （GMA） shown in Table 2 in the manuscript, run the following command: 


```bash
#cora
#run_metattack_rate-hang+stage2.py --dataset cora --function transformer --block constant_frac --lr 0.005 --dropout 0.4 --input_dropout 0.4 --time 4 --hidden_dim 64 --step_size 1 --runtime 10 --gpu 5 --epochs 800 --patience 100 --method predictor --alpha_ode 0.6  --ckpt_path [ '/stage-first-basegnn+ori/model_best_4_14_0.05.pth','/stage-first-basegnn+ori/robo_model_best_1_15_0.05.pth']

```
Note: --ckpt_path: the best checkpoint in stage-1 model without the fractional Lyapunov stability module.
