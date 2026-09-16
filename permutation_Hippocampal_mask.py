#!/usr/bin/env python

import os
import sys
import glob
import time
import pickle
import argparse
import warnings
from collections import defaultdict

import numpy as np
import pandas as pd

from nilearn.image import (load_img, concat_imgs, mean_img, new_img_like, resample_to_img, index_img, math_img)
from nilearn.maskers import NiftiMasker
from sklearn.svm import LinearSVC
from sklearn.model_selection import GroupKFold, permutation_test_score
from sklearn.metrics import roc_auc_score
from sklearn.preprocessing import LabelBinarizer
from sklearn.exceptions import ConvergenceWarning
from joblib import Parallel, delayed


parser = argparse.ArgumentParser()
parser.add_argument("--n_jobs", type=int, default=48, help="Number of parallel jobs (set to --cpus-per-task)")
parser.add_argument("--n_permutations", type=int, default=1000, help="Number of permutations per ROI")
parser.add_argument("--n_array_tasks", type=int, default=6, help="Total number of SLURM array tasks (must match --array upper bound + 1)")
args = parser.parse_args()

n_jobs         = args.n_jobs
n_permutations = args.n_permutations
n_array_tasks  = args.n_array_tasks


if "SLURM_ARRAY_TASK_ID" not in os.environ:
    print("WARNING: SLURM_ARRAY_TASK_ID not set — defaulting to 0 (local test mode)", flush=True)
    array_index = 0
else:
    array_index = int(os.environ["SLURM_ARRAY_TASK_ID"])

print(f"array task: {array_index} / {n_array_tasks - 1}", flush=True)
print(f"n_jobs: {n_jobs}", flush=True)
print(f"Permutations: {n_permutations}", flush=True)


def suppress_warnings():
    warnings.filterwarnings("ignore", category=ConvergenceWarning)

suppress_warnings()
Parallel(n_jobs=n_jobs)(delayed(suppress_warnings)() for _ in range(2 * n_jobs))


#LinearSVC does not support predict_proba, so we cannot use sklearn's built-in roc_auc_ovr scorer. Instead, we define a custom scorer that uses decision_function.

def roc_auc_ovr_scorer(estimator, X, y):

    decision_scores = estimator.decision_function(X)   
    lb = LabelBinarizer()
    y_bin = lb.fit_transform(y)                        
    return roc_auc_score(
        y_bin,
        decision_scores,
        multi_class="ovr"
    )


base_dir = os.getcwd()
output_dir = os.path.join(base_dir, "Masks")
perm_pkl = os.path.join(output_dir, "permutation_inputs.pkl")

if not os.path.exists(perm_pkl):
    sys.exit(
        f"ERROR: {perm_pkl} not found.\n"
        "run setup.py first and ensure it completed without errors."
    )

print(f"\nloading permutation inputs from:\n  {perm_pkl}", flush=True)
t0 = time.time()
with open(perm_pkl, "rb") as f:
    data = pickle.load(f)
print(f"loaded in {time.time()-t0:.1f}s", flush=True)

x_f_AB = data["x_f_AB"]
x_f_PD = data["x_f_PD"]
y_AB = data["y_AB"]
y_PD = data["y_PD"]

groups_AB = data["groups_AB"]
groups_PD = data["groups_PD"]
n_splits = data["n_splits"]

cross_task_folds = data["cross_task_folds"]

print(f"AB images : {x_f_AB.shape[-1]}", flush=True)
print(f"PD images : {x_f_PD.shape[-1]}", flush=True)

print(f"AB subjects : {len(np.unique(groups_AB))}", flush=True)
print(f"PD subjects : {len(np.unique(groups_PD))}", flush=True)
print(f"n_splits (CV) : {n_splits}",flush=True)
print(f"cross-task folds : {len(cross_task_folds)}",flush=True)  


#I need to have all betamaps to create a mean image to resample the AAL3 atlas to the same space as the betamaps.
betamaps  = os.path.join(base_dir, "folder", "sub-**", "**_run-**_**.nii.gz")
betafiles = glob.glob(betamaps)
print(f"  Found {len(betafiles)} beta files", flush=True)

t0 = time.time()
x_files = concat_imgs(betafiles)
m_img = mean_img(x_files, n_jobs=n_jobs)

aal_maps= load_img("/home/AAL3/AAL3v1.nii.gz")
aal_resampled=resample_to_img(aal_maps,m_img,interpolation="nearest", copy_header=True)

index = pd.read_csv("/home/AAL3/AAL3v1.nii.txt", sep = " ", names= ["Index", "Region", "Label"] ,  index_col=0 )
#index.iloc[1][2]

h = index.loc[(index["Region"] == "Hippocampus_L"), "Label"].values[0]
h2 = index.loc[(index["Region"] == "Hippocampus_R"), "Label"].values[0]

aal_resampled_d = aal_resampled.get_fdata()            #Basically in the mask, the regions are represented by the indices values #print(np.shape(aal_resampled_d)) - (58, 69, 53)
print (np.max(aal_resampled_d), h, h2)
hippo_mask_L = (aal_resampled_d == h).astype(np.uint8) #astype(np.uint8) converts numpy array of [True, False..] into [1,0..]
hippo_mask_R = (aal_resampled_d == h2).astype(np.uint8)
hippo_mask_combined = np.logical_or(hippo_mask_L, hippo_mask_R).astype(np.uint8)

hippo_map_L=new_img_like(aal_resampled, hippo_mask_L)
hippo_map_R=new_img_like(aal_resampled, hippo_mask_R)
hippo_map_combined =new_img_like(aal_resampled, hippo_mask_combined)


masks  = [hippo_map_L, hippo_map_R]
masks_names = ["hippocampus_L", "hippocampus_R"]

print(f"hippo masks are made, no of hippo masks {len(masks)}, names are {masks_names}")


#Within group permutation function
def run_permutation(mask_img, mask_name, X, y, groups, n_permutations, n_splits):


    masker = NiftiMasker(mask_img=mask_img)
    X_masked = masker.fit_transform(X)

    svc = LinearSVC(penalty="l2", max_iter=2000)
    cv = GroupKFold(n_splits=n_splits)      #GroupKFold instead of cv=5

    t_start = time.time()
    score, perm_scores, pval = permutation_test_score(
        svc,
        X_masked,
        y,
        groups=groups,                       #grouped at subject level
        cv=cv,                              
        scoring=roc_auc_ovr_scorer,
        n_permutations=n_permutations,
        n_jobs=1,                           #n_jobs=1 here because parallelism is handled by the outer Parallel call.
        verbose=0
    )

    elapsed = time.time() - t_start

    print(
        f"Done | roc_auc_ovr={score:.4f} | p={pval:.4f} | "
        f"time={elapsed:.1f}s | X={X_masked.shape}",
        flush=True
    )

    return {
        "name" : mask_name,
        "roc_auc_ovr": round(float(score), 5),
        "pvalue" : round(float(pval),  5),
        "elapsed" : f"{elapsed:.2f}s",
        "shape_X" : X_masked.shape,
        "shape_y" : y.shape,
    }

#cross task permutation function

def run_cross_task_permutation(mask_img, mask_name, x_f_AB, y_AB, x_f_PD, y_PD, cross_task_folds, n_permutations, rng_seed):
    
    print(f"[CrossTask] {mask_name}", flush=True)

    #feature extraction
    masker = NiftiMasker(mask_img=mask_img)
    X_AB_full = masker.fit_transform(x_f_AB)   #(n_AB_samples, n_voxels)
    X_PD_full = masker.transform(x_f_PD)       #(n_PD_samples, n_voxels)

    #true scores
    true_pd2ab_folds = []
    true_ab2pd_folds = []

    for fold in cross_task_folds:               #fold is created in such a way: same 36-train & 4-test participant 
        pd_tr = fold["pd_train_idx"]
        pd_te = fold["pd_test_idx"]
        ab_te = fold["ab_test_idx"]
        ab_tr = fold["ab_train_sub_idx"]

        svc1 = LinearSVC(penalty="l2", max_iter=2000)
        svc1.fit(X_PD_full[pd_tr], y_PD[pd_tr])
        true_pd2ab_folds.append(roc_auc_ovr_scorer(svc1, X_AB_full[ab_te], y_AB[ab_te]))

        svc2 = LinearSVC(penalty="l2", max_iter=2000)
        svc2.fit(X_AB_full[ab_tr], y_AB[ab_tr])
        true_ab2pd_folds.append(roc_auc_ovr_scorer(svc2, X_PD_full[pd_te], y_PD[pd_te]))

    true_pd2ab = float(np.mean(true_pd2ab_folds))
    true_ab2pd = float(np.mean(true_ab2pd_folds))

    #permutation distribution #training labels only changed, test data is unchanged.
    rng = np.random.default_rng(rng_seed)
    perm_scores_pd2ab = []
    perm_scores_ab2pd = []

    t_start = time.time()

    for _ in range(n_permutations):
        fold_pd2ab = []
        fold_ab2pd = []

        for fold in cross_task_folds:
            pd_tr = fold["pd_train_idx"]
            pd_te = fold["pd_test_idx"]
            ab_te = fold["ab_test_idx"]
            ab_tr = fold["ab_train_sub_idx"]

            #permute PD training labels
            y_perm_pd = rng.permutation(y_PD[pd_tr])
            svc1 = LinearSVC(penalty="l2", max_iter=2000)
            svc1.fit(X_PD_full[pd_tr], y_perm_pd)
            fold_pd2ab.append(roc_auc_ovr_scorer(svc1, X_AB_full[ab_te], y_AB[ab_te]))

            #permute AB training labels #sampled set
            y_perm_ab = rng.permutation(y_AB[ab_tr])
            svc2 = LinearSVC(penalty="l2", max_iter=2000)
            svc2.fit(X_AB_full[ab_tr], y_perm_ab)
            fold_ab2pd.append(roc_auc_ovr_scorer(svc2, X_PD_full[pd_te], y_PD[pd_te]))

        perm_scores_pd2ab.append(float(np.mean(fold_pd2ab)))
        perm_scores_ab2pd.append(float(np.mean(fold_ab2pd)))

    elapsed = time.time() - t_start

    #p-values: proportion of permutation scores >= true score 
    perm_arr_pd2ab = np.array(perm_scores_pd2ab)
    perm_arr_ab2pd = np.array(perm_scores_ab2pd)

    pval_pd2ab = (np.sum(perm_arr_pd2ab >= true_pd2ab) + 1) / (n_permutations + 1)
    pval_ab2pd = (np.sum(perm_arr_ab2pd >= true_ab2pd) + 1) / (n_permutations + 1)

    print(
        f"Done | PD to AB: roc_auc={true_pd2ab:.4f} p={pval_pd2ab:.4f} | "
        f"AB to PD: roc_auc={true_ab2pd:.4f} p={pval_ab2pd:.4f} | "
        f"time={elapsed:.1f}s",
        flush=True
    )

    return {
        "name" : mask_name,
        "PD_to_AB_roc_auc" : round(true_pd2ab, 5),
        "AB_to_PD_roc_auc" : round(true_ab2pd, 5),
        "PD_to_AB_pvalue" : round(float(pval_pd2ab), 5),
        "AB_to_PD_pvalue" : round(float(pval_ab2pd), 5),
        "elapsed" : f"{elapsed:.2f}s",
        "shape_X_AB" : X_AB_full.shape,
        "shape_X_PD" : X_PD_full.shape,
    }


#to run within group permutations for each group (AB and PD) separately
group_names = ["AB", "PD"]                                  
analysisfile = [(x_f_AB, y_AB, groups_AB), (x_f_PD, y_PD, groups_PD)]

all_results = defaultdict(dict)  


for group_name, (X, y, groups) in zip(group_names, analysisfile):
    print(f"\n{'-' * 50}", flush=True)
    print(f"within-group permutations: {group_name}", flush=True)
    print(f"{'-' * 50}", flush=True)

    t_group = time.time()

    parallel_results = Parallel(n_jobs=n_jobs)(
        delayed(run_permutation)(
            masks[i],
            masks_names[i],
            X,
            y,
            groups,
            n_permutations,
            n_splits
        )
        for i in range(len(masks))
    )

    #collect results
    result = {}
    for i, res in enumerate(parallel_results):
        result[res["name"]] = {
            "roc_auc_ovr" : res["roc_auc_ovr"],
            "pvalue" : res["pvalue"],
            "mask_type" : res["condition"],
            "time_taken" : res["elapsed"],
            "shape_info" : f"X={res['shape_X']}, y={res['shape_y']}",
        }

    all_results[group_name] = result

    #save per-group, per-array-task Excel (unique filename to no write conflicts)
    df = pd.DataFrame.from_dict(result, orient="index")
    df.insert(0, "Group", group_name)
    df.reset_index(inplace=True)
    df.rename(columns={"index": "ROI"}, inplace=True)

    excel_name = f"permutation_results_{group_name}_array{array_index}.xlsx"
    excel_path = os.path.join(base_dir, excel_name)
    df.to_excel(excel_path, index=False)
    print(f"\nsaved: {excel_path}",  flush=True)
    print(f"Group {group_name} total time: {time.time()-t_group:.1f}s", flush=True)


#to run cross-task permutations

print(f"\n{'-' * 50}", flush=True)
print("cross-task permutations", flush=True)
print(f"{'-' * 50}", flush=True)

t_cross = time.time()

#cross-task permutation in parallel for each ROI

cross_task_parallel_results = Parallel(n_jobs=n_jobs)(
    delayed(run_cross_task_permutation)(
        masks[i],
        masks_names[i],
        x_f_AB, y_AB,
        x_f_PD, y_PD,
        cross_task_folds,
        n_permutations,
        rng_seed=array_index * 100_000 + i   #unique seed per ROI
    )
    for i in range(len(masks))
)

#collect cross-task results
cross_task_result = {}
for res in cross_task_parallel_results:
    cross_task_result[res["name"]] = {
        "PD_to_AB_roc_auc" : res["PD_to_AB_roc_auc"],
        "AB_to_PD_roc_auc" : res["AB_to_PD_roc_auc"],
        "PD_to_AB_pvalue" : res["PD_to_AB_pvalue"],
        "AB_to_PD_pvalue" : res["AB_to_PD_pvalue"],
        "mask_type" : res["condition"],
        "time_taken" : res["elapsed"],
        "shape_info" : f"X_AB={res['shape_X_AB']}, X_PD={res['shape_X_PD']}",
    }

all_results["cross-task"] = cross_task_result

#save cross-task results to Excel for this array task
df_ct = pd.DataFrame.from_dict(cross_task_result, orient="index")
df_ct.insert(0, "direction", "PD to AB / AB to PD")
df_ct.reset_index(inplace=True)
df_ct.rename(columns={"index": "ROI"}, inplace=True)

ct_excel_name = f"permutation_results_CrossTask_array{array_index}.xlsx"
ct_excel_path = os.path.join(base_dir, ct_excel_name)
df_ct.to_excel(ct_excel_path, index=False)
print(f"\nsaved: {ct_excel_path}", flush=True)
print(f"cross-task total time: {time.time()-t_cross:.1f}s", flush=True)

#saving pickle of results
pkl_name = f"permutation_results_array{array_index}.pkl"
pkl_path = os.path.join(output_dir, pkl_name)
with open(pkl_path, "wb") as f:
    pickle.dump(dict(all_results), f)
print(f"\nPickle saved: {pkl_path}", flush=True)

print(f"permutation TASK {array_index} COMPLETE", flush=True)

