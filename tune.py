#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
    By using this code you can tune the trackers parameters with Optuna framework
    Attention: You should have TrackEval repository on your local machine, 
               then modify the pathes in this code.
    @author: STRH
    Created on Jul 30
"""

# import libraries
import os
import optuna
import argparse
import numpy as np
import pandas as pd
from pathlib import Path

# import tracker
from sort import Sort
from track import main

# running and evaluating tracker for every trials
def run_tracker(dataset, args, tracker):
    directories = [d for d in os.listdir(dataset) if os.path.isdir(os.path.join(dataset, d))]
    directories.sort()

    directories = [item for item in directories if 'UAV' in item]

    for path in directories:
        args.input_path = os.path.join(dataset, f"{path}/img1")
        main(args= args)

def run_validation(args, tracker, trial_num):
    dataset = "/home/setare/Vision/Work/Tracking/Swarm Dataset/Drone Swarm/UAVSwarm-dataset/UAVSwarm-dataset-test"
    if not os.path.exists(os.path.join(dataset, "sort/data")):
        os.makedirs(os.path.join(dataset, "sort/data"))
    args.save_path = os.path.join(dataset, "sort/data")

    run_tracker(dataset, args, tracker)
    dataset_root = "/".join(dataset.split("/")[:-1])
    dataset_for_command = "\ ".join(dataset_root.split(" "))
    eval_command = f"python /home/setare/Vision/Work/Evaluation/TrackingEvaluation/TrackEval/scripts/run_mot_challenge.py \
        --GT_FOLDER {dataset_for_command} --TRACKERS_FOLDER {dataset_for_command} \
            --OUTPUT_FOLDER {dataset_for_command} --TRACKERS_TO_EVAL sort \
                --BENCHMARK UAVSwarm-dataset --SPLIT_TO_EVAL test"
    
    os.system(eval_command)
    
    result = os.path.join(dataset_root, "sort/pedestrian_detailed.csv")
    result_csv = pd.read_csv(result)

    HOTA = result_csv.at[26,'HOTA___AUC']
    MOTA = result_csv.at[26,'MOTA']
    MOTP = result_csv.at[26,'MOTP']
    IDF1 = result_csv.at[26,'IDF1']

    if not os.path.exists(os.path.join(dataset_root,"sort_trials")):
        os.mkdir(os.path.join(dataset_root,"sort_trials"))
    result_csv.to_csv(f"{dataset_root}/sort_trials/trial_{trial_num}.csv", encoding='utf-8', index=False)

    return HOTA, MOTA, MOTP, IDF1

def objective(trial):
    # Example: Replace with your actual evaluation code
    # dummy_metric = your_evaluation_function(track_thresh, track_buffer, match_thresh)
    max_age = trial.suggest_int('max_age', 10, 50)
    min_hits = trial.suggest_int('min_hits', 1, 10)
    iou_threshold = trial.suggest_float('iou_threshold', 0, 1)
    
    # Initialize the tracker with these parameters
    tracker = Sort(
        max_age=max_age,
        min_hits=min_hits,
        iou_threshold=iou_threshold
    )

    HOTA, MOTA, MOTP, IDF1 =  run_validation(args, tracker=tracker, trial_num=trial.number)
    score = HOTA * 0.7 + MOTA * 0.1 + MOTP * 0.1 + IDF1 * 0.1
    return score

    # Return the metric to be optimized (minimized or maximized)
    # return dummy_metric

if __name__ == "__main__":
    # Set fixed arguments once at the beginning
    parser = argparse.ArgumentParser(description="Multi-object Tracking Inference to Evaluation")
    parser.add_argument('--weights', type=str, required=True, help="Path to detection model weights")
    parser.add_argument('--input_type', type=str, required=True, help="Input type: image or video")
    parser.add_argument('--input_path', type=str, required=False, help="Path to input images folder or video file")
    parser.add_argument('--save_mot', type=str, required=True, help="save results in mot format")
    parser.add_argument('--save_path', type=str, required=False, help="path to folder for saving results")
    parser.add_argument('--gt', type=str, required=False, help="path to gt.txt file")
    parser.add_argument('--save_video', type=str, required=True, help="if you want to save the tracking result visualization set it True")

    # Parse the fixed arguments
    args = parser.parse_args()

    study = optuna.create_study(study_name= "sort", storage= "sqlite:///sort.db", direction="maximize", load_if_exists=True)  # Choose "maximize" or "minimize"
    study.optimize(objective, n_trials=100)

    best_params = study.best_params
    print("Best Hyperparameters:", best_params)