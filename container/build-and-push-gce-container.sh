#!/bin/zsh

export LOCATION="us-west1"
export REPOSITORY_NAME="deepcell-benchmarking"
export PROJECT=$(gcloud config list project --format "value(core.project)")
export IMAGE_NAME="${LOCATION}-docker.pkg.dev/${PROJECT}/${REPOSITORY_NAME}/benchmarking:gce"

docker build . -t $IMAGE_NAME
docker push $IMAGE_NAME
