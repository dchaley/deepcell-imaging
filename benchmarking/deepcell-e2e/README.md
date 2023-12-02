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
  - Run [the notebook](deepcell-e2e-benchmark.ipynb), which documents the configuration parameters
  - Set the notebook runtime to `tensorflow 2.12 (local)`
 
NOTE: You must set up the model info (create/copy)
  - at ` /home/jupyter/.keras/models/MultiplexSegmentation`

Tip: To manually install DeepCell, run `pip install deepcell --user` in the VertexAI *terminal* prior to running the test notebook
