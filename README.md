# DeepCell-imaging

<img src="https://github.com/dchaley/deepcell-imaging/blob/main/images/deepcell-perf.png" width=1000>

This working Repo contains our notes / utilities / info for our cloud [DeepCell](https://github.com/vanvalenlab/deepcell-tf) imaging project. 

## Goal and Key Links

- GOAL: Understand and optimize using DeepCell to perform cellular image segmentation on GCP at scale.
- KEY LINK #1: Here's a link to our example/testing [notebooks](notebooks).
- KEY LINK #2: See our [project board](https://github.com/users/dchaley/projects/1) to understand our work areas for this project.

## Notes

### Python vs reading tiffs
- Alpineer (Angelo Lab helper for deepcell) [loads TIFFs from disk](https://github.com/angelolab/alpineer/blob/4e1bb1a0f96876f7ee8bdba4ec8bdf1b826e740f/src/alpineer/load_utils.py#L177) using io.imread.
  - io is an alias for `skimage.io`, [scikit-image](https://scikit-image.org/)
  - This has a `tifffile_v3` [plugin](https://imageio.readthedocs.io/en/stable/_autosummary/imageio.plugins.tifffile_v3.html), which wraps the `tifffile` library ([github](https://github.com/cgohlke/tifffile)).
- Tifffile is "the implementation" for reading tiffs from the filesystem.
  - We are investigating exactly how it reads files, e.g. does it extract metadata then read the sub-images (channels etc) on demand or does it read the whole file in all cases?
