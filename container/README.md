# DeepCell benchmarking container

This sets up a container that executes the steps in the benchmark setup notebook. It checks out the repo so in principle we could run the notebook however we haven't tried that yet.

Followed [Google instructions on derivative containers](https://cloud.google.com/deep-learning-containers/docs/derivative-container).

To build/push the container,

```
export LOCATION="us-west1"
export REPOSITORY_NAME="deepcell-benchmarking"
export PROJECT=$(gcloud config list project --format "value(core.project)")
export IMAGE_NAME="${LOCATION}-docker.pkg.dev/${PROJECT}/${REPOSITORY_NAME}/benchmarking:v3"
```

We need to change the image name version tag each build/push.

# Command to build * push docker image

```
gcloud artifacts repositories create $REPOSITORY_NAME \
    --repository-format=docker \
    --location=$LOCATION
gcloud auth configure-docker ${LOCATION}-docker.pkg.dev

docker build . -t $IMAGE_NAME
docker push $IMAGE_NAME
```

# Command to create Vertex AI custom job w/ container

Available machine types: [docs](https://cloud.google.com/vertex-ai/docs/training/configure-compute#machine-types).

Available accelerator types: [docs](https://cloud.google.com/vertex-ai/docs/reference/rest/v1/MachineSpec#AcceleratorType).

Use this for t4 gpu: `export GPU_TYPE="NVIDIA_TESLA_T4"`

```
export JOB_NAME="Benchmark n1-highmem-4 yyyymmddhhmmss"
export MACHINE_TYPE="n1-highmem-4"
export GPU_COUNT=0
export GPU_TYPE=""
export CONTAINER_URI="us-west1-docker.pkg.dev/deepcell-401920/deepcell-benchmarking/benchmarking@sha256:ed12549aab1c201042234a7e4fd79237dfdbd4257156e239abeef2de84994fae"
export REPLICA_COUNT=1

gcloud ai custom-jobs create \
  --region=$LOCATION --project=$PROJECT \
  --display-name="$JOB_NAME" \
  --worker-pool-spec=machine-type="$MACHINE_TYPE",accelerator-type="$GPU_TYPE",accelerator-count="$GPU_COUNT",replica-count="$REPLICA_COUNT",container-image-uri=$CONTAINER_URI --args="--custom_job_name='$JOB_NAME'"
```
