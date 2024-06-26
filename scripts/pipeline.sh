#!/bin/sh

tmp_preprocess_output="gs://deepcell-batch-jobs_us-central1/job-runs/tmp-pipeline/preprocessed.npz.gz"
tmp_predict_output="gs://deepcell-batch-jobs_us-central1/job-runs/tmp-pipeline/raw_predictions.npz.gz"
tmp_postprocess_output="gs://deepcell-batch-jobs_us-central1/job-runs/tmp-pipeline/postprocessed.npz.gz"

input_png_uri="gs://deepcell-batch-jobs_us-central1/job-runs/tmp-pipeline/input.png"
predictions_png_uri="gs://deepcell-batch-jobs_us-central1/job-runs/tmp-pipeline/predictions.png"

python scripts/preprocess.py --image_uri $1 --output_uri $tmp_preprocess_output
python scripts/predict.py --image_uri $tmp_preprocess_output --output_uri $tmp_predict_output
python scripts/postprocess.py --raw_predictions_uri $tmp_predict_output --output_uri $tmp_postprocess_output --input_rows 512 --input_cols 512
python scripts/visualize.py --image_uri $1 --predictions_uri $tmp_postprocess_output --visualized_input_uri $input_png_uri --visualized_predictions_uri $predictions_png_uri


