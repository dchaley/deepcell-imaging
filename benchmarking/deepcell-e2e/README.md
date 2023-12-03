# End-to-end DeepCell benchmark

**Objective**: measure how long it takes to process an image, varying the image size & resource allocation.

(define processing here; tldr: load input channels + run predict)

(insert some benchmarking tables and charts here)

## Test data provenance

See also: [sample data README](https://github.com/dchaley/deepcell-imaging/tree/main/sample-data)

# Method

- Provision a GCP Vertex AI managed notebook instance
  - Connect to the instance using the link `OPEN JUPYTERLAB`
- Upload the example notebook
  - Run [the notebook](deepcell-e2e-benchmark.ipynb), which installs dependencies documents the configuration parameters
    - If you install dependencies this way, you need to restart the kernel after the pip install.
    - Just run, restart, and re-run – it should be fine.
  - Set the notebook runtime to `tensorflow 2.12 (local)`
    - But something is off with the kernel? See also: https://github.com/dchaley/deepcell-imaging/issues/59
 
Tip: to avoid restarting the kernel after installing dependencies, run: `pip install --user -r requirements.txt` in the VertexAI *terminal* prior to running the benchmark notebook.
