FROM us-docker.pkg.dev/deeplearning-platform-release/gcr.io/tf2-gpu.2-8.py37

# The google special container is:
# us-docker.pkg.dev/vertex-ai-restricted/prediction/tf_opt-gpu.2-8:latest
#
# See also:
# https://cloud.google.com/vertex-ai/docs/predictions/optimized-tensorflow-runtime#available_container_images


ADD https://api.github.com/repos/dchaley/deepcell-imaging/git/refs/heads/main version.json

RUN git clone https://github.com/dchaley/deepcell-imaging.git

WORKDIR "/deepcell-imaging"

RUN pip install --user --upgrade --quiet -r requirements.txt

ENTRYPOINT ["python", "benchmarking/deepcell-e2e/benchmark.py"]
