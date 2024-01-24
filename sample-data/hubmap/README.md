# HuBMAP sample data

HuBMAP: [Human BioMolecular Atlas Program](https://portal.hubmapconsortium.org/)

Data is shared according to [HuBMAP external data sharing policy](https://hubmapconsortium.org/policies/external-data-sharing-policy/).

⚠️ note: many apps and/or devices struggle with multi-hundred megabyte TIFF images. You may need specialized software such as [QuPath](https://qupath.github.io/) to visualize the input TIFFs. May the force be with you. 🫡

## hbm873.pcpz.247.904M-px

[HuBMAP data page](https://portal.hubmapconsortium.org/browse/dataset/beb1b65624fe85b527ee2ce80ef208b2)

> Channels:
>
> - DAPI (excitation: 335-383 nm/emission: 420-470 nm)
> - eGFP (excitation: 450-490 nm/emission: 500-550 nm)
> - dsRed (excitation: 538-562 nm/emission: 570-640 nm)
>
> Resolution: 0.65 x 0.65 um

⚠️ Caution: resolution is 0.65 um; deepcell expects 0.5 um. Not handling for now in postprocessing.

Input image:

- 25836 x 34980, 8-bit grayscale, 3 channels, 4.5 GB
- `gs://davids-genomics-data-public/cellular-segmentation/hubmap/hbm873.pcpz.247.904M-px/VAN0054-LK-3-31-preAF-registered.ome.tiff`

TIFF file processed by [notebook](https://github.com/dchaley/deepcell-imaging/blob/480100fba876f3169dda86e45b35cc35d5d6a492/notebooks/Extract-Sample_hbm873.pcpz.247.904M-px.ipynb) into input_channels.tgz file: `gs://davids-genomics-data-public/cellular-segmentation/hubmap/hbm873.pcpz.247.904M-px/input_channels.npz`

TODO: add sample images