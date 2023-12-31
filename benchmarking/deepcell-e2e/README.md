# End-to-end DeepCell benchmark

**Objective**: measure how long it takes to process an image, varying the image size & resource allocation.

(define processing here; tldr: load input channels + run predict)

(insert some benchmarking tables and charts here)

## Test data provenance

See also: [sample data README](https://github.com/dchaley/deepcell-imaging/tree/main/sample-data)

# Method
## set up
- Provision a GCP Vertex AI managed notebook instance
  - Connect to the instance using the link `OPEN JUPYTERLAB`
  - set idle time to 10 min
  - Take note of the notebook runtime ID (see below to enter into the notebook)
    
- Clone the repo
    - in jupyter, go to the github icon, and click "clone a repository"
    - ![Screenshot 2023-12-06 at 9 22 57 PM](https://github.com/dchaley/deepcell-imaging/assets/352005/8424a47e-ab92-47dd-ab2e-365c3f1e4448)

    - repository url: https://github.com/dchaley/deepcell-imaging.git
 
- Set the notebook runtime to `tensorflow 2.10 (local)`
    - But something is off with the kernel? See also: https://github.com/dchaley/deepcell-imaging/issues/59

- Installing dependencies
  - Run [the setup notebook](setup.ipynb), which installs dependencies.
    
## Running the benchmark
- adjust the machine type as needed
- adjust the input_channels_path parameter to the location of the input image
- Adjust the parameter: `notebook_runtime_id`. Use the "Notebook name" from the workbench list. <img width="538" alt="Screenshot 2023-12-06 at 9 53 37 AM" src="https://github.com/dchaley/deepcell-imaging/assets/352005/07ed4f96-5dc7-44e8-ae5f-ec6e12d3c244">

- run the whole part.
- cp & paste the output to the csv file that records the benchmark
- record the benchmark results in https://docs.google.com/spreadsheets/d/1mHeKZqoH0-XrrV_P0uTwXNrwQCkcM6lbTZKGmMYKb8k/edit#gid=0 
