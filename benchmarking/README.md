
# Benchmarking deepcell imaging

# Idea

Deploy cloud function, configure its memory/CPU ([see docs](https://cloud.google.com/functions/docs/configuring/memory))

Run some benchmarking, maybe some warm-ups then measure "A Bunch" of calls to the function. (*Question*: in parallel? in sequence?)

Assign tags as appropriate, maybe by benchmark setup: machine size (ram/cpu settings), input size (pixels? file MBs? yes?)

# Notes

## Starting point

[This notebook](https://github.com/angelolab/ark-analysis/blob/main/templates/1_Segment_Image_Data.ipynb) by Angelo Lab shows how to convert from some tiff files into deepcell input then run deepcell

Potential contribution? Separate code to convert from img to numpy, then from numpy to tiffs

We are working with deep cell in memory … so not sure we need indirection to/from image files (it is *very useful* for proof of work though)

## How does Cloud Function do files, even

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

