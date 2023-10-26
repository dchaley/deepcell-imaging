# deepcell-imaging
Notes / utilities / info for Deep Cell imaging project. Our goal is to understand and optimize using DeepCell to perform cellular image segmentation most effectively on GCP.  

You are probably interested in the [notebooks](notebooks).

See our [project board](https://github.com/users/dchaley/projects/1) to understand our work areas for this project.

## Notes

### Python vs reading tiffs

Alpineer (Angelo Lab helper for deepcell) [loads TIFFs from disk](https://github.com/angelolab/alpineer/blob/4e1bb1a0f96876f7ee8bdba4ec8bdf1b826e740f/src/alpineer/load_utils.py#L177) using io.imread.

io is an alias for `skimage.io`, [scikit-image](https://scikit-image.org/)

This has a `tifffile_v3` [plugin](https://imageio.readthedocs.io/en/stable/_autosummary/imageio.plugins.tifffile_v3.html), which wraps the `tifffile` library ([github](https://github.com/cgohlke/tifffile)).

Tifffile seems to be "the implementation" for reading tiffs from the filesystem.

I'm not sure how it reads files right now, e.g. does it extract metadata then read the sub-images (channels etc) on demand or does it read the whole file no matter what?
