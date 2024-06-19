#!/bin/bash


#########################################
# NOTE: WE DID NOT FINISH TESTING THIS! #
#########################################

# Normally we'd use uuidgen, but it isn't always available.
JOB_UUID=$(python -c "import uuid; print(uuid.uuid4())")
JOB_NAME="gce-benchmark-$JOB_UUID"

INPUT_NPZ_PATH="gs://davids-genomics-data-public/cellular-segmentation/deep-cell/vanvalenlab-multiplex-20200810_tissue_dataset/mesmer-sample-11/input_channels.npz"

OUTPUT_PATH="gs://deepcell-batch-jobs_us-central1/job-runs/$JOB_NAME"

DOCKER_IMAGE="us-central1-docker.pkg.dev/deepcell-on-batch/deepcell-benchmarking-us-central1/benchmarking:gce"
MACHINE_TYPE="n1-standard-8"
GPU="nvidia-tesla-t4"
GPU_COUNT=1
PROVISIONING_MODEL="SPOT"
ZONE="us-central1-a"

gcloud compute instances create-with-container $JOB_NAME \
  --container-image $DOCKER_IMAGE \
  --container-restart-policy never \
  --container-arg="--input_channels_path='$INPUT_NPZ_PATH'" \
  --container-arg="--output_path='$OUTPUT_PATH'" \
  --container-arg="--visualize_input" \
  --container-arg="--visualize_predictions" \
  --container-arg="--provisioning_model=spot" \
  --machine-type $MACHINE_TYPE \
  --accelerator="type=$GPU,count=$GPU_COUNT" \
  --provisioning-model=$PROVISIONING_MODEL \
  --metadata startup-script="gs://deepcell-batch-jobs_us-central1/gce-driver-install-startup-script.sh" \
  --zone $ZONE

echo Submitted GCE VM: $JOB_NAME
