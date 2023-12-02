# End-to-end DeepCell benchmark

**Objective**: measure how long it takes to process an image, varying the image size & resource allocation.

(define processing here; tldr: load input channels + run predict)

(insert some benchmarking tables and charts here)

## Test data provenance

See also: [sample data README](https://github.com/dchaley/deepcell-imaging/tree/main/sample-data)

# Method

- Setup a GCP Vertex AI managed notebook instance, connect to the instance using the link `OPEN JUPYTERLAB`
- Run `pip install deepcell` in the VertexAI *terminal* prior to running the test notebook
- Run [the notebook](deepcell-e2e-benchmark.ipynb), which documents the configuration parameters
