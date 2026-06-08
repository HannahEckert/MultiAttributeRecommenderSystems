
import argh

import pandas as pd
import numpy as np
from tqdm import tqdm
from argh import arg
import os

def prepare_post_processing(top_k_path, artist_metadata_path, listening_events_path):

    top_k = pd.read_csv(top_k_path, sep='\t')
    artist_metadata = pd.read_csv(artist_metadata_path)
    listening_events_big = pd.read_csv(listening_events_path)

    #normalize scores by user
    top_k['normalized_score'] = top_k.groupby('user_id')['score'].transform(lambda x: x / x.sum())

    #listening_events_big = listening_events_big.merge(artist_metadata, left_on="artist", right_on="artist_name", how="left")

    if "synthetic" not in top_k_path:

        item_artis_matching = listening_events_big[["track_id", "artist_id"]].drop_duplicates()

        top_k = top_k.merge(item_artis_matching, left_on="item_id", right_on="track_id", how="left")

        top_k = top_k.merge(artist_metadata[["artist_id", "country", "gender"]], left_on="artist_id", right_on="artist_id", how="left")

        return listening_events_big, top_k
    
    else:

        top_k = top_k.merge(artist_metadata, left_on="item_id", right_on="track_id", how="left")

        return listening_events_big, top_k


def trade_off_method(listening_events_big, top_k, l=0.25, target_distribution = "interactions", version = "sum", synthetic = False):

    #create target distributions
    if target_distribution == "interactions":

        if not synthetic:
            target_distribution_countries = listening_events_big.groupby("country").size() / listening_events_big.shape[0]
            target_distribution_gender = listening_events_big.groupby("gender").size() / listening_events_big.shape[0]

        else:
            target_distribution_A = listening_events_big.groupby("attributeA").size() / listening_events_big.shape[0]
            target_distribution_B = listening_events_big.groupby("attributeB").size() / listening_events_big.shape[0]
            target_distribution_C = listening_events_big.groupby("attributeC").size() / listening_events_big.shape[0]

    elif target_distribution in ("societal", "population"):
        country_statistics = pd.read_csv("dataset/country_statistics.csv")
        target_distribution_countries = pd.Series(country_statistics.set_index("country")["population proportion"].to_dict())
        target_distribution_gender = pd.Series({"Non-binary":0.01, "Female":0.495, "Male":0.495})

    else:
        NotImplementedError("not a valid target distribution")


    if not synthetic:
        representations_countries = target_distribution_countries * 0
        representations_gender = target_distribution_gender * 0
    else:
        representations_A = target_distribution_A * 0
        representations_B = target_distribution_B * 0
        representations_C = target_distribution_C * 0
    
    number_scores_total = 0

    

    for user in tqdm(top_k["user_id"].unique(), desc="Re-ranking users"):

        user_top_k = top_k[top_k["user_id"] == user]

        if number_scores_total == 0:
            top_k_reranked = user_top_k.sort_values("normalized_score", ascending=False).reset_index(drop=True).head(10)
            top_k_reranked["rank"] = range(1, len(top_k_reranked) + 1)
            results = top_k_reranked

        else:
            if version =="sum":
                if not synthetic:
                    new_scores = (1 - l) * user_top_k["normalized_score"].to_numpy() + l * (
                        user_top_k["country"].map(representation_deficit_country).fillna(0).to_numpy()
                        +  user_top_k["gender"].map(representation_deficit_gender).fillna(0).to_numpy())
                else:
                    new_scores = (1 - l) * user_top_k["normalized_score"].to_numpy() + l * (
                        user_top_k["attributeA"].map(representation_deficit_A).fillna(0).to_numpy()
                        +  user_top_k["attributeB"].map(representation_deficit_B).fillna(0).to_numpy()
                        +  user_top_k["attributeC"].map(representation_deficit_C).fillna(0).to_numpy())
            elif version == "product":
                if not synthetic:
                    new_scores = user_top_k["normalized_score"].to_numpy() * (1+l*user_top_k["country"].map(representation_deficit_country).fillna(0).to_numpy()) * (1+l*user_top_k["gender"].map(representation_deficit_gender).fillna(0).to_numpy())
                else:
                    new_scores = user_top_k["normalized_score"].to_numpy() * (1+l*user_top_k["attributeA"].map(representation_deficit_A).fillna(0).to_numpy()) * (1+l*user_top_k["attributeB"].map(representation_deficit_B).fillna(0).to_numpy()) * (1+l*user_top_k["attributeC"].map(representation_deficit_C).fillna(0).to_numpy())
            else:
                raise ValueError(f"Version {version} not recognized")

            user_top_k = user_top_k.copy()
            user_top_k["new_scores"] = new_scores
            top_k_reranked = user_top_k.sort_values("new_scores", ascending=False).reset_index(drop=True).head(10)
            top_k_reranked["rank"] = range(1, len(top_k_reranked) + 1)
            results = pd.concat([results, top_k_reranked], ignore_index=True)

        if not synthetic:
            for country in top_k_reranked["country"].dropna().unique():
                representations_countries[country] += np.sum(1/np.log2(top_k_reranked[top_k_reranked["country"] == country]["rank"]+1))

            for gender in top_k_reranked["gender"].dropna().unique():
                representations_gender[gender] += np.sum(1/np.log2(top_k_reranked[top_k_reranked["gender"] == gender]["rank"]+1))
        else:
            for attributeA in top_k_reranked["attributeA"].dropna().unique():
                representations_A[attributeA] += np.sum(1/np.log2(top_k_reranked[top_k_reranked["attributeA"] == attributeA]["rank"]+1))

            for attributeB in top_k_reranked["attributeB"].dropna().unique():
                representations_B[attributeB] += np.sum(1/np.log2(top_k_reranked[top_k_reranked["attributeB"] == attributeB]["rank"]+1))

            for attributeC in top_k_reranked["attributeC"].dropna().unique():
                representations_C[attributeC] += np.sum(1/np.log2(top_k_reranked[top_k_reranked["attributeC"] == attributeC]["rank"]+1))

        number_scores_total += np.sum(1/np.log2(top_k_reranked["rank"]+1))

        if not synthetic:
            representation_deficit_country = np.log(target_distribution_countries / (representations_countries/number_scores_total)).replace([np.inf, -np.inf], np.nan).fillna(0)
            representation_deficit_gender = np.log(target_distribution_gender / (representations_gender/number_scores_total)).replace([np.inf, -np.inf], np.nan).fillna(0)
        else:
            representation_deficit_A = np.log(target_distribution_A / (representations_A/number_scores_total)).replace([np.inf, -np.inf], np.nan).fillna(0)
            representation_deficit_B = np.log(target_distribution_B / (representations_B/number_scores_total)).replace([np.inf, -np.inf], np.nan).fillna(0)
            representation_deficit_C = np.log(target_distribution_C / (representations_C/number_scores_total)).replace([np.inf, -np.inf], np.nan).fillna(0)

    return results





def baseline_marras(listening_events_big, top_k, l=0.25, target_distribution = "interactions", synthetic = False):

    #create target distributions
    if synthetic:
        if target_distribution == "interactions":
            target_distribution = (
                listening_events_big.groupby(["attributeA", "attributeB", "attributeC"]).size()
                / listening_events_big.shape[0]
            )
        else:
            raise NotImplementedError("Only 'interactions' target is supported for synthetic data")
    else:
        if target_distribution == "interactions":
            target_distribution = listening_events_big.groupby(["country", "gender"]).size() / listening_events_big.shape[0]
        elif target_distribution in ("societal", "population"):
            country_statistics = pd.read_csv("dataset/country_statistics.csv")
            pop_target_country = pd.Series(country_statistics.set_index("country")["population proportion"].to_dict())
            pop_target_gender  = pd.Series({"Non-binary": 0.01, "Female": 0.495, "Male": 0.495})
            index = pd.MultiIndex.from_product(
                [pop_target_country.index, pop_target_gender.index], names=["country", "gender"]
            )
            target_distribution = pd.Series(
                [pop_target_country[c] * pop_target_gender[g] for c, g in index],
                index=index,
            )
            target_distribution = target_distribution / target_distribution.sum()
        else:
            raise NotImplementedError("not a valid target distribution")

    # Build a group index mapping -> integer position in target array
    group_index = {key: i for i, key in enumerate(target_distribution.index)}
    G = len(target_distribution)
    sqrt_target = np.sqrt(target_distribution.values)  # shape (G,) — computed once

    results = []

    for user in tqdm(top_k["user_id"].unique(), desc="Re-ranking users"):
        user_top_k = top_k[top_k["user_id"] == user].reset_index(drop=True)

        # Map each candidate to its group index (-1 if unknown)
        if synthetic:
            candidate_groups = np.array([
                group_index.get((row["attributeA"], row["attributeB"], row["attributeC"]), -1)
                for _, row in user_top_k.iterrows()
            ], dtype=np.int32)
        else:
            candidate_groups = np.array([
                group_index.get((row["country"], row["gender"]), -1)
                for _, row in user_top_k.iterrows()
            ], dtype=np.int32)
        relevance = user_top_k["normalized_score"].to_numpy()

        # Cumulative discounted exposure vector over groups
        current_exposure = np.zeros(G, dtype=np.float64)
        selected_mask = np.zeros(len(user_top_k), dtype=bool)

        for position in range(1, 11):
            discount = 1.0 / np.log2(position + 1)
            candidate_mask = ~selected_mask
            if not candidate_mask.any():
                break

            # --- Vectorized Hellinger over all candidates at once ---
            # hyp_exposure[c] = current_exposure + discount * one_hot(group[c])
            # shape: (N_cands, G)
            hyp_exposure = np.tile(current_exposure, (candidate_mask.sum(), 1))
            cand_indices = np.where(candidate_mask)[0]
            cand_groups = candidate_groups[cand_indices]

            # Add discount only for candidates with a known group
            known = cand_groups >= 0
            hyp_exposure[known, cand_groups[known]] += discount

            totals = hyp_exposure.sum(axis=1, keepdims=True)
            totals[totals == 0] = 1.0  # avoid division by zero
            hyp_dist = hyp_exposure / totals  # (N_cands, G)

            # Squared Hellinger: 0.5 * sum((sqrt(target) - sqrt(hyp))^2, axis=1)
            hellinger_sq = 0.5 * np.sum((sqrt_target - np.sqrt(hyp_dist)) ** 2, axis=1)

            scores = (1 - l) * relevance[cand_indices] - l * hellinger_sq
            best_local = np.argmax(scores)
            best_idx = cand_indices[best_local]

            # Commit chosen item
            selected_mask[best_idx] = True
            g = candidate_groups[best_idx]
            if g >= 0:
                current_exposure[g] += discount

            chosen = user_top_k.iloc[best_idx].copy()
            chosen["rank"] = position
            results.append(chosen)

    results = pd.DataFrame(results)
    return results


def mitigation_continent(listening_events_big, top_k, l=1.0, target_distribution="interactions",
                          reranking_type="exposure", k=10, synthetic = False):
    """
    Implementation of the mitigationContinent algorithm (Deldjoo et al.)
    using intersectional (country, gender) groups.

    reranking_type : "visibility" (count-based) or "exposure" (discounted-exposure-based)
    k              : top-k cutoff (default 10)
    l              : fraction of possible swaps to consider at most (0 < l <= 1.0).
                     e.g. l=0.1 means at most 10% of candidate swaps are applied.
    """

    if synthetic:
        def get_group(row):
            a, b, c = row.get("attributeA"), row.get("attributeB"), row.get("attributeC")
            if pd.isna(a) or pd.isna(b) or pd.isna(c):
                return None
            return (a, b, c)
    else:
        def get_group(row):
            """Return (country, gender) tuple or None if either is missing."""
            c, g = row.get("country"), row.get("gender")
            if pd.isna(c) or pd.isna(g):
                return None
            return (c, g)

    if synthetic:
        if target_distribution == "interactions":
            target_props = (
                listening_events_big.groupby(["attributeA", "attributeB", "attributeC"]).size()
                / listening_events_big.shape[0]
            )
        else:
            raise NotImplementedError("Only 'interactions' target is supported for synthetic data")
    elif target_distribution == "interactions":
        target_props = (
            listening_events_big.groupby(["country", "gender"]).size()
            / listening_events_big.shape[0]
        )
    elif target_distribution in ("societal", "population"):
        country_statistics = pd.read_csv("dataset/country_statistics.csv")
        pop_target_country = pd.Series(
            country_statistics.set_index("country")["population proportion"].to_dict()
        )
        pop_target_gender = pd.Series({"Non-binary": 0.01, "Female": 0.495, "Male": 0.495})
        idx = pd.MultiIndex.from_product(
            [pop_target_country.index, pop_target_gender.index], names=["country", "gender"]
        )
        target_props = pd.Series(
            [pop_target_country[c] * pop_target_gender[g] for c, g in idx], index=idx
        )
        target_props = target_props / target_props.sum()
    else:
        raise NotImplementedError()

    top_k_df = top_k[top_k["rank"] <= k].copy()
    top_k_df["exposure"] = 1 / np.log2(top_k_df["rank"] + 1)
    total_exposure   = top_k_df["exposure"].sum()
    total_visibility = float(len(top_k_df))

    top_k_df["_group"] = top_k_df.apply(get_group, axis=1)
    valid_top_k = top_k_df.dropna(subset=["_group"])

    if reranking_type == "visibility":
        proportions = valid_top_k.groupby("_group").size().astype(float) / total_visibility
    else:
        proportions = valid_top_k.groupby("_group")["exposure"].sum() / total_exposure

    # Align to full group index
    all_groups   = proportions.index.union(target_props.index)
    target_props = target_props.reindex(all_groups, fill_value=0)
    proportions  = proportions.reindex(all_groups, fill_value=0)
    # continent_list[g] > 0  → over-represented (advantaged)
    # continent_list[g] < 0  → under-represented (disadvantaged)
    continent_list = proportions - target_props

    working = top_k.copy()
    working["exposure"] = 1 / np.log2(working["rank"] + 1)
    working["_group"]   = working.apply(get_group, axis=1)

    possible_swaps = []

    for user_id, user_df in tqdm(working.groupby("user_id"), desc="Collecting swaps"):
        user_df      = user_df.sort_values("rank")
        user_top_k   = user_df[user_df["rank"] <= k].sort_values("rank", ascending=False)
        user_outside = user_df[user_df["rank"] > k].sort_values("rank")

        out_cands, in_cands = [], []

        for _, row in user_top_k.iterrows():
            grp = row["_group"]
            if grp is None:
                continue
            if continent_list.get(grp, 0) > 0:   # advantaged → candidate to remove
                out_cands.append(row)

        for _, row in user_outside.iterrows():
            grp = row["_group"]
            if grp is None:
                continue
            if continent_list.get(grp, 0) < 0:   # disadvantaged → candidate to insert
                in_cands.append(row)

        i_in, i_out = 0, len(out_cands) - 1
        while i_in < len(in_cands) and i_out >= 0:
            item_in  = in_cands[i_in]
            item_out = out_cands[i_out]
            loss = item_out["normalized_score"] - item_in["normalized_score"]
            possible_swaps.append({
                "user_id":  user_id,
                "idx_out":  item_out.name,
                "idx_in":   item_in.name,
                "grp_out":  item_out["_group"],
                "grp_in":   item_in["_group"],
                "rank_out": item_out["rank"],
                "rank_in":  item_in["rank"],
                "exp_out":  item_out["exposure"],
                "exp_in":   item_in["exposure"],
                "loss":     loss,
            })
            i_in  += 1
            i_out -= 1

    # Sort by loss ascending (minor loss first)
    possible_swaps.sort(key=lambda x: x["loss"])

    # Apply at most l-fraction of the candidate swaps
    max_swaps = max(1, int(len(possible_swaps) * l))
    possible_swaps = possible_swaps[:max_swaps]

    used_idx     = set()
    rank_updates = {}  # original df index → new rank

    for swap in possible_swaps:
        idx_out = swap["idx_out"]
        idx_in  = swap["idx_in"]

        if idx_out in used_idx or idx_in in used_idx:
            continue

        grp_out = swap["grp_out"]
        grp_in  = swap["grp_in"]

        # Re-check conditions with updated continent_list
        if continent_list.get(grp_out, 0) <= 0:
            continue   # no longer advantaged
        if continent_list.get(grp_in, 0) >= 0:
            continue   # no longer disadvantaged

        # Commit swap: exchange ranks
        rank_updates[idx_out] = swap["rank_in"]
        rank_updates[idx_in]  = swap["rank_out"]

        # Update proportions
        if reranking_type == "visibility":
            exp_diff = 1.0 / total_visibility
        else:
            exp_diff = (swap["exp_out"] - swap["exp_in"]) / total_exposure

        proportions[grp_out] -= exp_diff
        proportions[grp_in]  += exp_diff
        continent_list = proportions - target_props

        used_idx.add(idx_out)
        used_idx.add(idx_in)

    working["rank"] = working.apply(lambda row: rank_updates.get(row.name, row["rank"]), axis=1)
    result = working[working["rank"] <= k].copy().drop(columns=["exposure", "_group"], errors="ignore")
    return result







def baseline_nails(listening_events_big, top_k, l=0.5, target_distribution="interactions", synthetic=False):
    """
    NAILS (News Article Importance-based List Scoring) adapted to our setting.

    For each attribute dimension (country + gender, or attributeA + attributeB for synthetic):
    1. Compute target distribution p_star (interactions-based or population).
    2. Compute model distribution p_ei from global normalized score sums across all
       users and candidates — same principle as NAILS' p(Ei) computed over all impressions.
    3. Compute adjustment weight p_w[val] = p_star[val] / p_ei[val].
    4. Adjust per-item scores multiplicatively across attribute dimensions:
           new_score_i = score_i * prod_col( (1 - lambda) + lambda * p_w[col][val_i] )
    5. Re-rank top-10 per user by adjusted score.
    """

    if synthetic:
        attr_cols = ["attributeA", "attributeB", "attributeC"]
    else:
        attr_cols = ["country", "gender"]

    target_dists = {}
    for col in attr_cols:
        if target_distribution == "interactions":
            target_dists[col] = listening_events_big.groupby(col).size() / len(listening_events_big)
        elif target_distribution in ("societal", "population"):
            if synthetic:
                raise NotImplementedError("population target not supported for synthetic data")
            if col == "country":
                cs = pd.read_csv("dataset/country_statistics.csv")
                t = pd.Series(cs.set_index("country")["population proportion"].to_dict())
                target_dists[col] = t / t.sum()
            else:  # gender
                target_dists[col] = pd.Series({"Non-binary": 0.01, "Female": 0.495, "Male": 0.495})
        else:
            raise NotImplementedError(f"target_distribution '{target_distribution}' not supported")

    # Sum normalized_score per attribute value across all users & candidates,
    # then normalise — mirrors NAILS' p(Ei) = sum(scores of cat i) / n_samples.
    p_ei = {}
    for col in attr_cols:
        col_scores = top_k.groupby(col)["normalized_score"].sum()
        p_ei[col] = col_scores / col_scores.sum()

    p_w = {}
    for col in attr_cols:
        p_star = target_dists[col]
        all_vals = p_star.index.union(p_ei[col].index)
        p_star_a = p_star.reindex(all_vals, fill_value=0.0)
        p_ei_a   = p_ei[col].reindex(all_vals, fill_value=1e-9)
        p_w[col] = p_star_a / p_ei_a

    results = []
    for user in tqdm(top_k["user_id"].unique(), desc="NAILS re-ranking"):
        user_top_k = top_k[top_k["user_id"] == user].copy().reset_index(drop=True)

        adjusted = user_top_k["normalized_score"].copy()
        for col in attr_cols:
            weights = user_top_k[col].map(p_w[col]).fillna(1.0)
            adjusted = adjusted * ((1 - l) + l * weights)

        user_top_k["_adjusted_score"] = adjusted
        top_k_reranked = (
            user_top_k.sort_values("_adjusted_score", ascending=False)
            .head(10)
            .reset_index(drop=True)
        )
        top_k_reranked["rank"] = range(1, len(top_k_reranked) + 1)
        results.append(top_k_reranked)

    result_df = pd.concat(results, ignore_index=True).drop(columns=["_adjusted_score"], errors="ignore")
    return result_df


@arg('--top_k_path', type=str, default="experiments/lfm_small/output/top_k_valid.tsv")
@arg('--artist_metadata_path', type=str, default= "dataset/artists_metadata.csv")
@arg('--listening_events_path', type=str, default= "dataset/listening_events_filtered.csv")
@arg("--output_path", type=str, default= "experiments/lfm_small/output/post_processing")
@arg('--method', type=str, default="trade_off", help='Post-processing method to use')
@arg('--l', type=float, default=0.25, help='Trade-off parameter for the trade-off method')
@arg("--target_distribution", type=str,default = "interactions" )
@arg("--seed", type=int, default=42, help='Random seed for shuffling user order')
@arg("--version", type=str, default="sum", help="version of the trade off method to use (sum or product)")
@arg("--reranking_type", type=str, default="exposure", help="reranking type for mitigation_continent: visibility or exposure")

def post_processing(
    *,
    top_k_path="experiments/lfm_small/output/top_k.tsv",
    artist_metadata_path="dataset/artists_metadata.csv",
    listening_events_path="dataset/listening_events.csv",
    output_path="experiments/lfm_small/output/post_processing",
    method="trade_off",
    l=0.25,
    target_distribution="interactions",
    seed=42,
    version="sum",
    reranking_type="exposure",
):

    listening_events_big, top_k = prepare_post_processing(top_k_path, artist_metadata_path, listening_events_path)

    # Shuffle user order with the given seed
    rng = np.random.default_rng(seed)
    user_order = top_k["user_id"].unique()
    user_order = rng.permutation(user_order)
    top_k = pd.concat([top_k[top_k["user_id"] == u] for u in user_order], ignore_index=True)

    if "synthetic" not in top_k_path:
        if method == "trade_off":
            results = trade_off_method(listening_events_big, top_k, l, target_distribution, version)
        elif method == "marras":
            results = baseline_marras(listening_events_big, top_k, l, target_distribution)
        elif method == "mitigation_continent":
            results = mitigation_continent(listening_events_big, top_k, l, target_distribution,
                                        reranking_type=reranking_type)
        elif method == "nails":
            results = baseline_nails(listening_events_big, top_k, l, target_distribution, synthetic=False)
        else:
            raise NotImplementedError(f"Method {method} not implemented")
        
    else:
        if method == "trade_off":
            results = trade_off_method(listening_events_big, top_k, l, target_distribution, version, synthetic=True)
        elif method == "marras":
            results = baseline_marras(listening_events_big, top_k, l, target_distribution, synthetic=True)
        elif method == "mitigation_continent":
            results = mitigation_continent(listening_events_big, top_k, l, target_distribution,
                                        reranking_type=reranking_type, synthetic=True)
        elif method == "nails":
            results = baseline_nails(listening_events_big, top_k, l, target_distribution, synthetic=True)
        else:
            raise NotImplementedError(f"Method {method} not implemented")
    
    #create output folder if it doesn't exist
    if not os.path.exists(output_path):
        os.makedirs(output_path)

    results.to_csv(os.path.join(output_path, "post_processed_results.csv"), index=False)



if __name__ == '__main__':
    argh.dispatch_command(post_processing)