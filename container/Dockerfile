# Base container image is the *rebuilt* tensorflow 2.8.4-gpu image.
# See also: https://github.com/dchaley/tensorflow-2.8.4-redux
FROM us-central1-docker.pkg.dev/deepcell-on-batch/deepcell-benchmarking-us-central1/base-tf-2.8.4-rebuilt-gpu:latest

# Refresh the key to nvidia's repositories.
# https://forums.developer.nvidia.com/t/notice-cuda-linux-repository-key-rotation/212771
RUN apt-key adv --fetch-keys https://developer.download.nvidia.com/compute/cuda/repos/ubuntu1804/x86_64/3bf863cc.pub

# Update installed packages & install graphviz & git
# Clear out the aptitude cache as well.
# Lastly, install and/or update pip.
RUN apt-get update && \
    apt-get install -y graphviz git && \
    rm -rf /var/lib/apt/lists/* && \
    /usr/bin/python3 -m pip install --no-cache-dir --upgrade pip

# Add the repo sha to the container as the version.
ADD https://api.github.com/repos/dchaley/deepcell-imaging/git/refs/heads/main version.json

# Clone the deepcell-imaging repo
RUN git clone https://github.com/dchaley/deepcell-imaging.git

# Switch into the repo directory
WORKDIR "/deepcell-imaging"

# Install python requirements
RUN pip install --user --upgrade -r requirements.txt

# The container entrypoint is the benchmark script.
# Command-line arguments go to the script.
ENTRYPOINT ["python", "benchmarking/deepcell-e2e/benchmark.py"]
