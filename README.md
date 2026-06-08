# MultiAttributeRecommenderSystems

step 0: put lfm dataset or any dataset you want to use in the experiments/DATASET/input folder or skip this step if you want to use the synthetic dataset already in the folder

step 1: train baseline model with train.py. 

Example: python train.py synthetic -m BPR -c config/recbole_config_default.yaml -k 10 -o experiments/synthetic/outputBPR42 -s 42 --clean

step 2: run post-processing method + baselines with run_post_provessing_parallel.py. 

Example (our method is trade_off the others are baselines): python run_post_processing_parallel.py --model BPR --dataset synthetic --method trade_off marras mitigation_continent nails --seeds 1 



# Appendix:

## Additional Plots

Following is a complete collection of fairness-utility trade-off plots, as not all target distributions, dataset, backbone model combinations made it into the final paper. We report NDCG@10 against MSDE for intersectional and attribute-wise provider groups. Points indicate different fairness weights, and black stars indicate the unmodified backbone model.

# BPR, LFM2b
<img width="1180" height="599" alt="image" src="https://github.com/user-attachments/assets/a59e4cf3-d67f-4f04-9f25-0849a9b549a1" />


