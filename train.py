
import argparse
import json
import logging
import shutil
from pathlib import Path

import argh
import numpy as np
import pandas as pd
from argh import arg
from recbole.config import Config
from recbole.quick_start import load_data_and_model
from tqdm import tqdm

from recbole_wrapper import run_recbole_experiment, get_recbole_scores, get_recbole_scores_valid, get_recbole_ndcg_per_user

import torch

EXPERIMENTS_FOLDER = Path('experiments')

def prepare_run(dataset_name: str, model: str, clean=False, output_name='output', seed=None):
    """
    Prepares dataset folder structure and asserts all required files are present.
    Uses a unique working directory per (dataset, model, seed) tuple so multiple runs
    can execute in parallel without clobbering each other's recbole_tmp/ and saved/ dirs.
    :returns: (tracks_file, work_dir) where work_dir is the unique run directory
    """
    experiment_folder = EXPERIMENTS_FOLDER / dataset_name
    if not experiment_folder.exists():
        raise FileNotFoundError(f'Dataset {dataset_name} not found')

    if not (experiment_folder / 'input').exists():
        raise FileNotFoundError(f'Dataset invalid: Input folder missing')

    tracks_file = experiment_folder / 'input' / 'listening_events_recbole.csv'
    if not tracks_file.exists():
        raise FileNotFoundError(f'Dataset invalid: input/listening_events_recbole.csv file missing')

    # Unique working directory per (dataset, model, seed) to allow parallel runs
    seed_suffix = f'_seed{seed}' if seed is not None else ''
    work_dir = Path(f'recbole_tmp_{dataset_name}_{model}{seed_suffix}')
    saved_dir = work_dir / 'saved'

    shutil.rmtree(work_dir, ignore_errors=True)
    (work_dir / 'dataset').mkdir(exist_ok=True, parents=True)
    saved_dir.mkdir(exist_ok=True, parents=True)

    # Convert CSV to tab-separated .inter file for RecBole
    inter_dest = work_dir / 'dataset' / 'dataset.inter'
    pd.read_csv(tracks_file).to_csv(inter_dest, sep='\t', index=False)

    if clean:
        shutil.rmtree(experiment_folder / 'datasets', ignore_errors=True)
        shutil.rmtree(experiment_folder / output_name, ignore_errors=True)
        shutil.rmtree(experiment_folder / 'log', ignore_errors=True)
    (experiment_folder / 'datasets').mkdir(exist_ok=True)
    (experiment_folder / output_name).mkdir(exist_ok=True)
    (experiment_folder / 'log').mkdir(exist_ok=True)

    return tracks_file, work_dir


def cleanup(dataset_name: str):

    # remove the saved, log, log_tensorboard and recbole_workdir folder
    #shutil.rmtree('saved', ignore_errors=True) # We want to keep the saved model for later analysis, so we don't delete it here
    shutil.rmtree('log', ignore_errors=True)
    shutil.rmtree('log_tensorboard', ignore_errors=True)


def compute_top_k_scores(scores, dataset: str, k=10, orig_user_ids=None, orig_item_ids=None):
    """
    Computes the top k scores per user and returns a dataframe with the new interactions.
    Uses GPU-accelerated torch.topk when a CUDA device is available, otherwise falls back
    to a vectorised numpy implementation. Both paths are fully batched (no Python loops
    over users).
    """
    n = len(scores)

    # Use vectorised numpy (argpartition) — the scores matrix is typically too
    # large to fit on GPU, and numpy is already fast enough for this operation.
    top_indices = np.argpartition(scores, -k, axis=1)[:, -k:]
    top_scores = np.take_along_axis(scores, top_indices, axis=1)
    order = np.argsort(-top_scores, axis=1)
    top_indices = np.take_along_axis(top_indices, order, axis=1)
    top_scores = np.take_along_axis(top_scores, order, axis=1)

    # Map internal column indices -> original item IDs
    if orig_item_ids is not None:
        orig_item_ids_arr = np.asarray(orig_item_ids)
        flat_item_ids = orig_item_ids_arr[top_indices.ravel()]
    else:
        flat_item_ids = top_indices.ravel()

    if orig_user_ids is None:
        orig_user_ids = np.arange(n)
    flat_user_ids = np.repeat(orig_user_ids, k)

    df = pd.DataFrame({
        'user_id': flat_user_ids,
        'item_id': flat_item_ids,
        'rank': np.tile(np.arange(1, k + 1), n),
        'score': top_scores.ravel(),
    })
    return df

def split_to_df(data_loader, dataset):
    rows = []
    uid2items = data_loader.dataset.uid2positive_item
    for uid_token in range(1, len(uid2items)):  # skip [PAD] at index 0
        items = uid2items[uid_token]
        if items is None:
            continue
        orig_user = int(dataset.id2token(dataset.uid_field, uid_token))
        orig_items = dataset.id2token(dataset.iid_field, items.numpy()).astype(int)
        for orig_item in orig_items:
            rows.append((orig_user, orig_item))
    return pd.DataFrame(rows, columns=['user_id', 'item_id'])

@arg('dataset_name', type=str)
@arg('-m', '--model', type=str, help='Name of RecBole model to be used')
@arg('-c', '--config', type=str, help='Path to the Recbole config file')
@arg('-k', type=int, help='Number of items to be recommended per user')
@arg('-o', '--output-name', type=str, help='Name of the output folder inside the experiment directory')
@arg('-s', '--seed', type=int, help='Random seed (overrides config file seed)')
@arg('--clean', action=argparse.BooleanOptionalAction)


def train(
        dataset_name, model='BPR', 
        config='recbole_config_default.yaml',
        k=10,
        output_name='output',
        seed=None,
        clean=False):

    tracks_path, work_dir = prepare_run(dataset_name, model=model, clean=clean, output_name=output_name, seed=seed)
    saved_dir = work_dir / 'saved'

    call_params = {
        'dataset_name': dataset_name,
        'model': model,
        'config': config,
    }
    with open(EXPERIMENTS_FOLDER / dataset_name / 'params.json', 'w') as f:
        json.dump(call_params, f, indent=2)

    config_dict = {'data_path': str(work_dir), 'checkpoint_dir': str(saved_dir)}
    if seed is not None:
        config_dict['seed'] = seed
    config = Config(model=model, dataset='dataset', config_file_list=[config],
                    config_dict=config_dict)


    run_recbole_experiment(model=model, dataset=dataset_name, config=config)


    # Attempt to make sure the model is garbage collected and doesn't leak memory
    del config

    # There should only be one model file in saved folder, get its path
    model_path = str(next(f for f in saved_dir.iterdir() if f.name.startswith(model)))
    config, model, dataset, train_data, valid_data, test_data = load_data_and_model(model_file=model_path)

    # Obtain recommendation scores
    #scores, orig_item_ids = get_recbole_scores(model, dataset, test_data, config)

    # Get original user token IDs in the same order as the rows in the scores matrix
    orig_user_ids = [int(float(k)) for k in dataset.field2token_id['user_id'].keys() if k != '[PAD]']

    """
    # Obtain top k scores and save them for later analysis
    top_k_df = compute_top_k_scores(scores, dataset_name, k=k, orig_user_ids=orig_user_ids, orig_item_ids=orig_item_ids)
    top_k_df.to_csv(
        EXPERIMENTS_FOLDER / dataset_name / 'output' / f'top_k.tsv', header=True,
        sep='\t', index=False)
    """

    # Also compute top_k masking train+test (valid items only), for comparison with ndcg_per_user.tsv
    (top_indices, top_scores_arr, orig_item_ids), _ = get_recbole_scores_valid(model, dataset, valid_data, test_data, config, k=k)
    orig_item_ids_arr = np.asarray(orig_item_ids)
    orig_user_ids_arr = np.asarray(orig_user_ids)
    top_k_valid_df = pd.DataFrame({
        'user_id': np.repeat(orig_user_ids_arr, k),
        'item_id': orig_item_ids_arr[top_indices.ravel()],
        'rank': np.tile(np.arange(1, k + 1), len(orig_user_ids_arr)),
        'score': top_scores_arr.ravel(),
    })
    top_k_valid_df.to_csv(
        EXPERIMENTS_FOLDER / dataset_name / output_name / f'top_k_valid.tsv', header=True,
        sep='\t', index=False)
    
    # Save train/val/test splits with original IDs
    output_folder = EXPERIMENTS_FOLDER / dataset_name / output_name
    
    # Convert test_data from RecBole internal IDs to original IDs
    recbole_to_orig_user = {v: int(float(k)) for k, v in dataset.field2token_id['user_id'].items() if k != '[PAD]'}
    recbole_to_orig_item = {v: int(float(k)) for k, v in dataset.field2token_id['item_id'].items() if k != '[PAD]'}

    test_rows = []
    for recbole_uid, test_items_tensor in enumerate(test_data.uid2positive_item[1:], start=1):
        if test_items_tensor is None:
            continue
        orig_uid = recbole_to_orig_user[recbole_uid]
        for iid in test_items_tensor:
            test_rows.append((orig_uid, recbole_to_orig_item[iid.item()]))

    test_df = pd.DataFrame(test_rows, columns=['user_id', 'item_id'])
    test_df.to_csv(output_folder / 'test.tsv', sep='\t', index=False)

    # Compute and save per-user NDCG@k, matching RecBole's internal evaluation pool
    ndcg_per_user, valid_df = get_recbole_ndcg_per_user(model, dataset, train_data, valid_data, test_data, config, k=k)
    ndcg_df = pd.DataFrame([{'user_id': uid, f'ndcg@{k}': val} for uid, val in ndcg_per_user.items()])
    ndcg_df.to_csv(
        EXPERIMENTS_FOLDER / dataset_name / output_name / f'ndcg_per_user.tsv',
        sep='\t', index=False
    )
    valid_df.to_csv(
        EXPERIMENTS_FOLDER / dataset_name / output_name / 'valid.tsv',
        sep='\t', index=False
    )

    # Move saved/ folder into the experiment output folder, then clean up work_dir
    saved_dest = EXPERIMENTS_FOLDER / dataset_name / output_name / 'saved'
    shutil.rmtree(saved_dest, ignore_errors=True)
    shutil.move(str(saved_dir), str(saved_dest))
    shutil.rmtree(work_dir, ignore_errors=True)


if __name__ == '__main__':
    argh.dispatch_command(train)