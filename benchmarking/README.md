
# Benchmarking deepcell imaging

Current benchmarks. See also: [2023-10-18 Deepcell performance optimization notes](https://docs.google.com/document/d/1LVOktQ9RAAn5jjYhDpH_OSoDLtSdHc0-3BcaclNnvO4/edit) and [2023-10-13 deepcell notes running on vertex ai](https://docs.google.com/document/d/1aMqO9H09AEsqOqfJ27byvRA3wmLuxdqy6kuwQMnTmhA/edit)

Time:

| Task        | Local | Cloud |
|-------------|-------|-------|
| Preprocess  | 53s   | 43s   |
| Predict     | 928s  | 193s  |
| Postprocess | 275s  | 264s  |

Resources:

| Resource | Local  | Cloud   |
|----------|--------|---------|
| CPU      | 6-core | 16 vcpu |
| GPU      | 0      | 1 T4    | 
| RAM      | 16     | 60      |

# Idea / raw notes 

Start with the "[notebook pipeline](../notebooks/README.md)".

Run a Vertex AI "execution" with various machine sizes. (I can't see any other way to programmatically run a notebook on Vertex AI? I don't wanna set up & measure the permutations myself but maybe am overthinking it)

Record timing results.
- testest
- Assign tags as appropriate, maybe by benchmark setup: machine size (ram/cpu settings), input size (pixels? file MBs? yes?)

# Notes

## Starting point

[This notebook](https://github.com/angelolab/ark-analysis/blob/main/templates/1_Segment_Image_Data.ipynb) by Angelo Lab shows how to convert from some tiff files into deepcell input then run deepcell

#### Potential Work Area
- Separate code to convert from `img` to `numpy`
- then from `numpy` to `tiffs`

NOTE: We are working with deep cell in memory … so not sure we need indirection to/from image files (it is *very useful* for proof of work though)

## Test Cloud Function 

Cloud Function local storage appears to be implemented in RAM, [see docs](https://cloud.google.com/functions/docs/concepts/execution-environment#memory-file-system)

> The function execution environment includes an in-memory file system [...] Use of the file system counts towards a function's memory usage.

Implications:

- we have to clean up downloaded files
- we can reuse (or have to reload) previously opened files
- file load aka networking is 💰 + ⏳

### Cloud Function Memory Info

From docs (and our testing):
- v1 vs v2 differing limits
- default for both is 256 MB
- v1 max is ~ 8 GB
- v2 max is ~ 16 GB

* Also - max CPU is 4 and no GPUs are available. v2 creates a 'visible' container is published to GCP Artifact registry and can be modified and/or used in CloudRun with has the capability of more RAM and CPU, still no GPUs.
* Additionally, max associated disk for CloudFunctions is 100 MB compressed or 500 MB for uncompressed.

