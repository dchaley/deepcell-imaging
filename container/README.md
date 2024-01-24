# DeepCell benchmarking container

This sets up a container that executes the steps in the benchmark setup notebook. It checks out the repo so in principle we could run the notebook however we haven't tried that yet.

Followed [Google instructions on derivative containers](https://cloud.google.com/deep-learning-containers/docs/derivative-container).

To build/push the container,

```
export LOCATION="us-west1"
export REPOSITORY_NAME="deepcell-benchmarking"

export PROJECT=$(gcloud config list project --format "value(core.project)")
gcloud artifacts repositories create $REPOSITORY_NAME \
    --repository-format=docker \
    --location=$LOCATION
gcloud auth configure-docker ${LOCATION}-docker.pkg.dev

export IMAGE_NAME="${LOCATION}-docker.pkg.dev/${PROJECT}/${REPOSITORY_NAME}/benchmarking:v1"
docker build . -t $IMAGE_NAME
docker push $IMAGE_NAME
```

I guess we change v1 to v2 on next push…?
