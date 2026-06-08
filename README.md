# MultiAttributeRecommenderSystems

step 0: put lfm dataset or any dataset you want to use in the experiments/DATASET/input folder or skip this step if you want to use the synthetic dataset already in the folder

step 1: train baseline model with train.py. Example: python train.py synthetic -m BPR -c config/recbole_config_default.yaml -k 10 -o experiments/synthetic/outputBPR42 -s 42 --clean

step 2: run post-processing method + baselines with run_post_provessing_parallel.py. Example (our method is trade_off the others are baselines): python run_post_processing_parallel.py --model BPR --dataset synthetic --method trade_off marras mitigation_continent nails --seeds 1 
