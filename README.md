# MultiAttributeRecommenderSystems

step 0: put lfm dataset or any dataset you want to use in the experiments/DATASET/input folder or skip this step if you want to use the synthetic dataset already in the folder

step 1: train baseline model with train.py. 

Example: python train.py synthetic -m BPR -c config/recbole_config_default.yaml -k 10 -o experiments/synthetic/outputBPR42 -s 42 --clean

step 2: run post-processing method + baselines with run_post_provessing_parallel.py. 

Example (our method is trade_off the others are baselines): python run_post_processing_parallel.py --model BPR --dataset synthetic --method trade_off marras mitigation_continent nails --seeds 1 



# Appendix:

## Additional Plots

Following is a complete collection of fairness-utility trade-off plots, as not all target distributions, dataset, backbone model combinations made it into the final paper. We report NDCG@10 against MSDE for intersectional and attribute-wise provider groups. Points indicate different fairness weights, and black stars indicate the unmodified backbone model.

### BPR, LFM2b
<img width="1180" height="599" alt="image" src="https://github.com/user-attachments/assets/a59e4cf3-d67f-4f04-9f25-0849a9b549a1" />

### NeuMF, LFM2b 
<img width="1180" height="599" alt="image" src="https://github.com/user-attachments/assets/378e9fe6-a9a1-40ef-b9b2-3f28f44061db" />

### LightGCN, LFM2b
<img width="1180" height="599" alt="image" src="https://github.com/user-attachments/assets/8e0e0fcb-65bc-4580-8c99-55ceb58dc19e" />

### synthetic, all models
<img width="1377" height="895" alt="synthetic" src="https://github.com/user-attachments/assets/1c382eb4-1be7-425c-838b-e81e84faa734" />


## Additional Training Details 
All models are trained three times with seeds 42,43,44. Probabilistic post-ptocessing methods are additionally run 5 times with seeds 1,2,3,4,5. Reported results are averegeed over runs. Specific training details are as reported in the config/recbole_config_default.yaml file. 






