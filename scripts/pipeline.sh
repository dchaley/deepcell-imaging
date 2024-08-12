#!/bin/sh

workdir="gs://deepcell-batch-jobs_us-central1/job-runs/tmp-pipeline"

tmp_preprocess_output="$workdir/preprocessed.npz.gz"
tmp_predict_output="$workdir/raw_predictions.npz.gz"
tmp_postprocess_output="$workdir/postprocessed.npz.gz"
tmp_tiff_output="$workdir/predictions.tiff"

input_png_uri="$workdir/input.png"
predictions_png_uri="$workdir/predictions.png"

tmp_preprocess_benchmark="$workdir/benchmark_preprocess.json"
tmp_predict_benchmark="$workdir/benchmark_preprocess.json"
tmp_postprocess_benchmark="$workdir/benchmark_preprocess.json"

# Classic model (SavedModel format)
# model_path="gs://genomics-data-public-central1/cellular-segmentation/vanvalenlab/deep-cell/vanvalenlab-tf-model-multiplex-downloaded-20230706/MultiplexSegmentation.tar.gz"
# model_hash="a1dfbce2594f927b9112f23a0a1739e0"

# New model (HDF5 format)
model_path="gs://genomics-data-public-central1/cellular-segmentation/vanvalenlab/deep-cell/vanvalenlab-tf-model-multiplex-downloaded-20230706/MultiplexSegmentation-resaved-20240710.h5"
model_hash="56b0f246081fe6b730ca74eab8a37d60"

python scripts/preprocess.py --image_uri $1 --output_uri $tmp_preprocess_output --benchmark_output_uri $tmp_preprocess_benchmark
python scripts/predict.py --image_uri $tmp_preprocess_output --model_path $model_path --model_hash $model_hash --output_uri $tmp_predict_output --benchmark_output_uri $tmp_predict_benchmark
python scripts/postprocess.py --raw_predictions_uri $tmp_predict_output --output_uri $tmp_postprocess_output --input_rows 512 --input_cols 512 --tiff_output_uri $tmp_tiff_output --benchmark_output_uri $tmp_postprocess_benchmark
python scripts/visualize.py --image_uri $1 --predictions_uri $tmp_postprocess_output --visualized_input_uri $input_png_uri --visualized_predictions_uri $predictions_png_uri


