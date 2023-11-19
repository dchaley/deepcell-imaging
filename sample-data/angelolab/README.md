# ARK Example data

Sample data sourced from the Angelo Lab's `ark_example` [dataset](https://huggingface.co/datasets/angelolab/ark_example). Specifically, [this version](https://huggingface.co/datasets/angelolab/ark_example/blob/8affc79/data/image_data.zip) of `image_data.zip`. The data was processed with the [Extract-ArkExample-Sample-Channels](../../notebooks/Extract-ArkExample-Sample-Channels.ipynb) notebook (after having run an ARK notebook).

All sample data is in Google Cloud Storage: `gs://davids-genomics-data-public/cellular-segmentation/deep-cell/angelolab-ark-example`

## ark-example-X

Each sample directory contains the following files.

1. `input.tiff`

A 2-page TIFF file, showing the nuclear & membrane channels respectively. Each page is the combination of several FOVs. See also the [1_Segment_Image_Data](https://github.com/angelolab/ark-analysis/blob/bfc53b7d6f7bdf96ef95dd03ea84f9dfefc74dc9/templates/1_Segment_Image_Data.ipynb) notebook.

Example: `gs://davids-genomics-data-public/cellular-segmentation/deep-cell/angelolab-ark-example/ark-example-0/input.tiff` ([url](https://storage.cloud.google.com/davids-genomics-data-public/cellular-segmentation/deep-cell/angelolab-ark-example/ark-example-0/input.tiff))

2. `input_channels.npz`

An npz archive containing the array `input_channels`.

```
In : np.load('sample-data/ark-example/ark-example-0/input_channels.npz')['input_channels'].shape
Out: (512, 512, 2)
```

The dimensions are: rows, columns, channels. There are two channels: nuclear & membrane, in that order.

Example: `gs://davids-genomics-data-public/cellular-segmentation/deep-cell/angelolab-ark-example/ark-example-0/input_channels.npz` ([url](https://storage.cloud.google.com/davids-genomics-data-public/cellular-segmentation/deep-cell/angelolab-ark-example/ark-example-0/input_channels.npz))

3. `predicted_segments.tiff`

The predicted segments as saved from the [1_Segment_Image_Data](https://github.com/angelolab/ark-analysis/blob/bfc53b7d6f7bdf96ef95dd03ea84f9dfefc74dc9/templates/1_Segment_Image_Data.ipynb) notebook.

Example: `gs://davids-genomics-data-public/cellular-segmentation/deep-cell/angelolab-ark-example/ark-example-0/predicted_segments.tiff` ([url](https://storage.cloud.google.com/davids-genomics-data-public/cellular-segmentation/deep-cell/angelolab-ark-example/ark-example-0/predicted_segments.tiff))
