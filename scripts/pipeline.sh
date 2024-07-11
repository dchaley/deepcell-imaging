#!/bin/sh

tmp_preprocess_output="gs://deepcell-batch-jobs_us-central1/job-runs/tmp-pipeline/preprocessed.npz.gz"
tmp_predict_output="gs://deepcell-batch-jobs_us-central1/job-runs/tmp-pipeline/raw_predictions.npz.gz"
tmp_postprocess_output="gs://deepcell-batch-jobs_us-central1/job-runs/tmp-pipeline/postprocessed.npz.gz"
tmp_tiff_output="gs://deepcell-batch-jobs_us-central1/job-runs/tmp-pipeline/predictions.tiff"

input_png_uri="gs://deepcell-batch-jobs_us-central1/job-runs/tmp-pipeline/input.png"
predictions_png_uri="gs://deepcell-batch-jobs_us-central1/job-runs/tmp-pipeline/predictions.png"

# Classic model (SavedModel format)
# model_path="gs://genomics-data-public-central1/cellular-segmentation/vanvalenlab/deep-cell/vanvalenlab-tf-model-multiplex-downloaded-20230706/MultiplexSegmentation.tar.gz"
# model_hash="a1dfbce2594f927b9112f23a0a1739e0"

# New model (Keras format)
model_path="gs://genomics-data-public-central1/cellular-segmentation/vanvalenlab/deep-cell/vanvalenlab-tf-model-multiplex-downloaded-20230706/MultiplexSegmentation-resaved-20240710.h5"
model_hash="56b0f246081fe6b730ca74eab8a37d60"

python scripts/preprocess.py --image_uri $1 --output_uri $tmp_preprocess_output
python scripts/predict.py --image_uri $tmp_preprocess_output --model_path $model_path --model_hash $model_hash --output_uri $tmp_predict_output
python scripts/postprocess.py --raw_predictions_uri $tmp_predict_output --output_uri $tmp_postprocess_output --input_rows 512 --input_cols 512 --tiff_output_uri $tmp_tiff_output
python scripts/visualize.py --image_uri $1 --predictions_uri $tmp_postprocess_output --visualized_input_uri $input_png_uri --visualized_predictions_uri $predictions_png_uri


