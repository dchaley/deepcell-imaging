#!/bin/bash

gcloud builds submit --region=us-central1 --tag us-central1-docker.pkg.dev/deepcell-on-batch/deepcell-benchmarking-us-central1/benchmarking:gce
