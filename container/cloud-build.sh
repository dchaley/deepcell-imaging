#!/bin/bash

if [[ -z "$GCP_ARTIFACT_REPOSITORY" ]]; then
  echo "Error: GCP_ARTIFACT_REPOSITORY is not set."
  exit 1
fi

if [ -f "Dockerfile" ]; then
  echo "Error: run from repository root."
  exit 1
fi

gcloud builds submit \
  --region=us-central1 \
  --config=container/build-batch-container.yaml \
  --substitutions _GCP_REPOSITORY=$GCP_ARTIFACT_REPOSITORY \
  .
