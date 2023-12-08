# 10x Genomics sample data

All data from 10x Genomics' [dataset resources](https://www.10xgenomics.com/resources/datasets).

⚠️ note: many apps and/or devices struggle with multi-hundred megabyte TIFF images. You may need specialized software such as [QuPath](https://qupath.github.io/) to visualize the input TIFFs. May the force be with you. 🫡

## preview-human-breast-20221103-418mb

[10x Genomics data page](https://www.10xgenomics.com/products/xenium-in-situ/preview-dataset-human-breast)

> - Channel 1 = 1:200 CD20, Goat anti-Mouse IgG-488 , (Abcam, Cat. No. ab219329, Thermo Fisher Cat. No. A-11029)
> - Channel 2 = 1:1500 HER2, Goat anti-Rabbit IgG-594, (Abcam Cat. No. ab134182, Abcam Cat. No. ab150088)
> - Channel 3 = DAPI (Thermo Fisher, Cat. No. 62248)

Input image:

- 14239  ×  9777, 8-bit grayscale, 3 channels, 418 MB
- `gs://davids-genomics-data-public/cellular-segmentation/10x-genomics/preview-human-breast-20221103-418mb/Xenium_FFPE_Human_Breast_Cancer_Rep1_if_image.tif`
- [direct url](https://storage.cloud.google.com/davids-genomics-data-public/cellular-segmentation/10x-genomics/preview-human-breast-20221103-418mb/Xenium_FFPE_Human_Breast_Cancer_Rep1_if_image.tif) (needs a google login)
- md5sum: `cf8e28717304d0490c0e763b9b4b07b2`

TIFF file processed by [notebook](https://github.com/dchaley/deepcell-imaging/blob/main/notebooks/Extract-Sample_preview-human-breast-20221103-418mb.ipynb) into input_channels.tgz file: `gs://davids-genomics-data-public/cellular-segmentation/10x-genomics/preview-human-breast-20221103-418mb/input_channels.npz`

Selected portion, approx 2300 × 1675; HER2 channel (channel #2 of 3). [Breastcancer.org](https://www.breastcancer.org/pathology-report/her2-status) says "HER2 proteins are receptors on breast cells". So (I think) this image shows the membranes (outer layers) of breast cells. ([full size image](preview-human-breast-20221103-418mb/selectedregion_input_image_channel_HER2_small.png))

![HER2 channel of input image](preview-human-breast-20221103-418mb/selectedregion_input_image_channel_HER2_small.png)

Same portion, with all membrane & nuclear channels combined. Blue is membrane and green is nucleus. ([full size image](preview-human-breast-20221103-418mb/selectedregion_deepcell_input_image_small.png))

![green/blue visualization of cells](preview-human-breast-20221103-418mb/selectedregion_deepcell_input_image_small.png)

Same portion, segments predicted by DeepCell: ([full size image](preview-human-breast-20221103-418mb/selectedregion_deepcell_predicted_segments.png))

![HER2 channel of input image](preview-human-breast-20221103-418mb/selectedregion_deepcell_predicted_segments_small.png)

## human-prostate-cancer-20210727-725mb

[10x Genomics data page](https://www.10xgenomics.com/resources/datasets/human-prostate-cancer-adjacent-normal-section-with-if-staining-ffpe-1-standard-1-3-0)

> - Channel 1 = 1:100 Iba1/AIF-1 (Cell Signalling, Cat. No. 36618S)
> - Channel 2 = 1:100 Vimentin (Cell Signalling, Cat. No. 9856S)
> - Channel 3 = 1:5000 DAPI (Thermo Scientific, Cat. No. 62248)

TIFF file processed by [notebook](https://github.com/dchaley/deepcell-imaging/blob/main/notebooks/Extract-Sample_human-prostate-cancer-20210727-725mb.ipynb) into input_channels.tgz file: `gs://davids-genomics-data-public/cellular-segmentation/10x-genomics/human-prostate-cancer-20210727-725mb/input_channels.npz`

Input image:

- 26624 × 25088, 8-bit grayscale, 3 channels, 725 MB

md5sum: `1df0f4b6477242161637b05d74a23067`

TODO: visualization previews: https://github.com/dchaley/deepcell-imaging/issues/84

## Note: naming convention

The samples don't have concise IDs, for example: `human-prostate-cancer-adjacent-normal-section-with-if-staining-ffpe-1-standard-1-3-0`

We need something brief, descriptive, and unique-ish.

We'll use: `<tissue type>-<publish date>-<tiff MBs>`

eg `human-prostate-cancer-20210727-725mb`

If multiple tissue types are published on the same date, we'll either revisit or rely on the filesize to disambiguate. 🤔
