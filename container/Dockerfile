# syntax=docker/dockerfile-upstream:master
# ^^ we need that for the variable substitution feature.

# Base container image is the *rebuilt* tensorflow 2.8.4-gpu image.
# See also: https://github.com/dchaley/tensorflow-2.8.4-redux
ARG GCP_LOCATION=us-central1
ARG GCP_PROJECT_ID=my-project
ARG GCP_REPOSITORY=my-repository
ARG BASE_IMAGE=${GCP_LOCATION}-docker.pkg.dev/${GCP_PROJECT_ID}/${GCP_REPOSITORY}/base-tf-2.8.4-rebuilt-gpu:11.8.0-cudnn8-runtime-ubuntu22.04
FROM ${BASE_IMAGE}

# The github repository to build.
ARG GITHUB_REPOSITORY

# Refresh the key to nvidia's repositories.
# https://forums.developer.nvidia.com/t/notice-cuda-linux-repository-key-rotation/212771
RUN apt-key adv --fetch-keys https://developer.download.nvidia.com/compute/cuda/repos/ubuntu1804/x86_64/3bf863cc.pub

# Update installed packages & install graphviz & git
# Clear out the aptitude cache as well.
# Lastly, install and/or update pip.
RUN echo "deb [signed-by=/usr/share/keyrings/cloud.google.gpg] https://packages.cloud.google.com/apt cloud-sdk main" | tee -a /etc/apt/sources.list.d/google-cloud-sdk.list && \
    curl https://packages.cloud.google.com/apt/doc/apt-key.gpg | gpg --dearmor -o /usr/share/keyrings/cloud.google.gpg && \
    apt-get update -y && \
    apt-get install -y --no-install-recommends git google-cloud-sdk google-cloud-cli pigz && \
    rm -rf /var/lib/apt/lists/* && \
    /usr/bin/python3 -m pip install --no-cache-dir --upgrade pip && \
    gcloud config set storage/parallel_composite_upload_enabled True

# Pull the requirements.txt file.
# FIXME: we should be using our local copy of the requirements file & source.
# The way it is now, we can only build from the main branch.
# Changing this means rearranging how we build, so skipping for now.
ADD https://raw.githubusercontent.com/${GITHUB_REPOSITORY}/refs/heads/main/requirements.txt requirements.txt

# Install python requirements
RUN pip install --user --upgrade -r requirements.txt

# Add the repo sha to the container as the version.
ADD https://api.github.com/repos/${GITHUB_REPOSITORY}/git/refs/heads/main version.json

# Clone the deepcell-imaging repo
RUN git clone https://github.com/${GITHUB_REPOSITORY}.git

# Switch into the repo directory
WORKDIR "${GITHUB_REPOSITORY#*/}"

# Install our own module
RUN pip install .

# The container entrypoint is the benchmark script.
# Command-line arguments go to the script.
ENTRYPOINT ["python", "benchmarking/deepcell-e2e/benchmark.py"]
