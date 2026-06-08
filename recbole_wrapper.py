import math
import sys

import numpy as np
from recbole.config import Config
from recbole.data import create_dataset, data_preparation, construct_transform
from recbole.utils import init_seed, init_logger, get_model, get_flops, set_color, get_trainer, get_environment
from logging import getLogger

from recbole.utils.case_study import full_sort_scores
from tqdm import trange
import torch

import os




def run_recbole_experiment(model: str, dataset: str,config: Config):
    """
    Initially we used recbole.quick_start.run_recbole() to run the RecBole models.
    However, this has many limitations and undesired behaviour and thus we implemented the function ourselves

    """
    init_seed(config["seed"], config["reproducibility"])

    # logger initialization
    init_logger(config)
    logger = getLogger()
    logger.info(sys.argv)
    logger.info(config)

    # initialize the dataset according to config
    dataset = create_dataset(config)
    logger.info(dataset)

    # dataset splitting. Test_data is always empty and thus ignored in our case
    logger.info('Preparing dataset')
    train_data, valid_data, test_data = data_preparation(config, dataset)
    logger.info('Done!')

    # model loading and initialization
    init_seed(config["seed"] + config["local_rank"], config["reproducibility"])
    model_class = get_model(config["model"])
    # instantiate the model
    model = model_class(config, train_data._dataset).to(config["device"])
    logger.info(model)

    transform = construct_transform(config)
    try:
        flops = get_flops(model, dataset, config["device"], logger, transform)
        logger.info(set_color("FLOPs", "blue") + f": {flops}")
    except Exception as e:
        logger.warning(f"FLOPs calculation skipped: {e}")

    # trainer loading and initialization
    trainer = get_trainer(config["MODEL_TYPE"], config["model"])(config, model)


    # model training
    best_valid_score, best_valid_result = trainer.fit(
        train_data, valid_data, saved=True, show_progress=config["show_progress"]
    )
    logger.info(set_color("best valid ", "yellow") + f": {best_valid_result}")







def _get_ids(dataset):
    """Recbole internally uses different IDs to ours, and this mapping allows us to properly process their recommendations"""
    user_ids = list(dataset.field2token_id['user_id'].keys())
    # [PAD] user
    user_ids.remove('[PAD]')
    user_ids = dataset.token2id(dataset.uid_field, user_ids)
    user_ids = user_ids.astype(np.int64)

    item_ids = list(dataset.field2token_id['item_id'].keys())
    # [PAD] item
    item_ids.remove('[PAD]')
    item_ids = dataset.token2id(dataset.iid_field, item_ids)
    item_ids = item_ids.astype(np.int64)

    return user_ids, item_ids





def _predict_scores_batched(model, user_ids_batch, all_item_ids, device, item_batch_size):
    """
    Compute model scores for a batch of users over all items, iterating over items
    in chunks to avoid OOM errors (important for models like NeuMF with large item sets).
    Returns a numpy array of shape (len(user_ids_batch), len(all_item_ids)).
    """
    n_users = len(user_ids_batch)
    n_items = len(all_item_ids)
    result = np.empty((n_users, n_items), dtype=np.float32)

    user_tensor = torch.tensor(user_ids_batch, dtype=torch.long, device=device)

    for j in range(math.ceil(n_items / item_batch_size)):
        it_start = j * item_batch_size
        it_end = min(n_items, (j + 1) * item_batch_size)
        item_chunk = all_item_ids[it_start:it_end]
        item_tensor = torch.tensor(item_chunk, dtype=torch.long, device=device)

        # Build interaction for every (user, item) pair in the chunk
        n_chunk = len(item_chunk)
        users_rep = user_tensor.repeat_interleave(n_chunk)   # (n_users * n_chunk,)
        items_rep = item_tensor.repeat(n_users)              # (n_users * n_chunk,)

        interaction = {model.USER_ID: users_rep, model.ITEM_ID: items_rep}
        from recbole.data.interaction import Interaction
        interaction = Interaction(interaction)
        interaction = interaction.to(device)

        with torch.no_grad():
            chunk_scores = model.predict(interaction)  # (n_users * n_chunk,)

        result[:, it_start:it_end] = chunk_scores.cpu().numpy().reshape(n_users, n_chunk)

    return result


def get_recbole_scores(model, dataset, test_data, config: Config, user_batch_size: int = 32, item_batch_size: int = 1024):
    user_ids, item_ids = _get_ids(dataset)

    scores = np.empty((len(user_ids), len(item_ids)), dtype=np.float32)

    device = torch.device(config["device"])

    # Check whether full_sort_scores can be used (memory-safe for small item sets)
    # or whether we must batch over items manually (e.g. NeuMF with large catalogs).
    use_batched = True  # always use batched to avoid OOM

    for i in trange(math.ceil(len(user_ids) / user_batch_size),
                    desc="Calculating Recommendation Scores",
                    dynamic_ncols=True, smoothing=0):
        u_start = i * user_batch_size
        u_end = min(len(user_ids), (i + 1) * user_batch_size)

        if use_batched:
            scores[u_start:u_end] = _predict_scores_batched(
                model, user_ids[u_start:u_end], item_ids, device, item_batch_size
            )
        else:
            # full_sort_scores returns scores over ALL internal item IDs
            full_batch_scores = full_sort_scores(
                user_ids[u_start:u_end], model, test_data, device=device
            ).cpu().numpy().astype(np.float32)  # shape: (batch_users, n_all_items)

            n_item_batches = math.ceil(len(item_ids) / item_batch_size)
            for j in range(n_item_batches):
                it_start = j * item_batch_size
                it_end = min(len(item_ids), (j + 1) * item_batch_size)
                item_batch = item_ids[it_start:it_end]
                scores[u_start:u_end, it_start:it_end] = full_batch_scores[:, item_batch]

    # set scores of test set items to -inf so they are never recommended
    for i, items in enumerate(test_data.uid2positive_item[1:]):
        if items is None:
            continue
        # -1 because RecBole uses 1-based indexing with a [PAD] item
        items = items.cpu().numpy() - 1
        scores[i, items] = -np.inf

    # Build mapping from column index -> original item ID
    orig_item_ids = np.array(
        [int(float(dataset.id2token(dataset.iid_field, iid))) for iid in item_ids],
        dtype=np.int64
    )

    return scores, orig_item_ids




def get_recbole_scores_valid(model, dataset, valid_data, test_data, config: Config, batch_size: int = 32, k: int = 10):
    """
    Calculates top-k scores masking train+test items, leaving only valid items rankable.
    Computes top-k on the fly per user batch to avoid materializing the full scores matrix.
    """
    user_ids, item_ids = _get_ids(dataset)
    n_users = len(user_ids)
    n_items = len(item_ids)

    orig_item_ids = np.array(
        [int(float(dataset.id2token(dataset.iid_field, iid))) for iid in item_ids],
        dtype=np.int64
    )

    # Precompute test item masks per user (RecBole uses 1-based item IDs)
    test_mask = [set(items.cpu().numpy() - 1) if items is not None else set() for items in test_data.uid2positive_item[1:]]

    device = torch.device(config['device'])

    all_top_indices = np.empty((n_users, k), dtype=np.int64)
    all_top_scores = np.empty((n_users, k), dtype=np.float32)

    for i in trange(math.ceil(n_users / batch_size), desc='Calculating Valid Recommendation Scores',
                    dynamic_ncols=True, smoothing=0):
        start = i * batch_size
        end = min(n_users, (i + 1) * batch_size)
        batch_scores = _predict_scores_batched(
            model, user_ids[start:end], item_ids, device, item_batch_size=1024
        )  # (batch, n_items)

        # Mask test items per user in the batch
        for j, mask in enumerate(test_mask[start:end]):
            if mask:
                batch_scores[j, list(mask)] = -np.inf

        # Top-k for this batch
        top_idx = np.argpartition(batch_scores, -k, axis=1)[:, -k:]
        top_sc = np.take_along_axis(batch_scores, top_idx, axis=1)
        order = np.argsort(-top_sc, axis=1)
        top_idx = np.take_along_axis(top_idx, order, axis=1)
        top_sc = np.take_along_axis(top_sc, order, axis=1)

        all_top_indices[start:end] = top_idx
        all_top_scores[start:end] = top_sc

    # Build full scores-like return: map top-k column indices to item IDs
    # Return a compact representation instead of the full matrix
    return (all_top_indices, all_top_scores, orig_item_ids), orig_item_ids


def get_recbole_ndcg_per_user(model, dataset, train_data, valid_data, test_data, config: Config, k: int = 10, batch_size: int = 32):
    """
    Computes per-user NDCG@k exactly as RecBole's internal evaluator does.

    full_sort_scores must be called with valid_data (not test_data) so that
    RecBole only masks training items internally — leaving validation items
    available for ranking, just like the internal validation evaluator does.

    Returns a dict mapping original user_id -> ndcg@k value.
    """
    user_ids, item_ids = _get_ids(dataset)
    n_users = len(user_ids)

    orig_item_ids = np.array(
        [int(float(dataset.id2token(dataset.iid_field, iid))) for iid in item_ids],
        dtype=np.int64
    )

    # Build reverse mappings: RecBole internal ID -> original ID
    recbole_to_orig_user = {v: int(float(key)) for key, v in dataset.field2token_id['user_id'].items() if key != '[PAD]'}
    recbole_to_orig_item = {v: int(float(key)) for key, v in dataset.field2token_id['item_id'].items() if key != '[PAD]'}

    # Precompute valid items per user
    valid_items_per_user = [
        set(recbole_to_orig_item[iid.item()] for iid in tensor) if tensor is not None else set()
        for tensor in valid_data.uid2positive_item[1:]
    ]

    device = torch.device(config['device'])

    ndcg_per_user = {}
    valid_data_rows = []

    for i in trange(math.ceil(n_users / batch_size), desc='Calculating NDCG scores',
                    dynamic_ncols=True, smoothing=0):
        start = i * batch_size
        end = min(n_users, (i + 1) * batch_size)
        batch_scores = _predict_scores_batched(
            model, user_ids[start:end], item_ids, device, item_batch_size=1024
        )  # (batch, n_items)

        for j in range(end - start):
            global_i = start + j
            recbole_uid = global_i + 1
            orig_uid = recbole_to_orig_user[recbole_uid]
            valid_items_orig = valid_items_per_user[global_i]

            if not valid_items_orig:
                continue

            for orig_iid in valid_items_orig:
                valid_data_rows.append((orig_uid, orig_iid))

            top_k_cols = np.argpartition(batch_scores[j], -k)[-k:]
            top_k_cols = top_k_cols[np.argsort(-batch_scores[j][top_k_cols])]
            top_k_orig = orig_item_ids[top_k_cols]

            dcg = sum(1.0 / np.log2(rank + 2) for rank, item in enumerate(top_k_orig) if item in valid_items_orig)
            ideal_len = min(len(valid_items_orig), k)
            idcg = sum(1.0 / np.log2(rank + 2) for rank in range(ideal_len))
            ndcg_per_user[orig_uid] = dcg / idcg if idcg > 0 else 0.0

    import pandas as pd
    valid_df = pd.DataFrame(valid_data_rows, columns=['user_id', 'item_id'])

    return ndcg_per_user, valid_df

