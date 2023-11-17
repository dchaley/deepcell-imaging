# DeepCell sample data

This data was created with the [Extract-Sample-Channels](../../notebooks/Extract-Sample-Channels.ipynb) notebook.

Each sample directory contains the following files.

* `input.png`

A green/blue image showing the nuclear & membrane channels respectively.

* `input_channels.npz`

An npz archive containing the array `input_channels`.

```
In : np.load('sample-data/deepcell/mesmer-sample-3/input_channels.npz')['input_channels'].shape
Out: (512, 512, 2)
```

The dimensions are: rows, columns, channels. There are two channels: nuclear & membrane, in that order.

* `input_rgb.npz`

An npz archive containing the array `rgb`: pixels.

```
In : np.load('sample-data/deepcell/mesmer-sample-3/input_rgb.npz')['rgb'].shape
Out: (512, 512, 3)
```

The dimensions are: rows, columns, R/G/B.

* `predicted_segments.png`

The input image, with segment predictions overlaid.

* `predictions.npz`

TODO: https://github.com/dchaley/deepcell-imaging/issues/25

* `raw_output_image.npz`

TODO: https://github.com/dchaley/deepcell-imaging/issues/25

* `raw_output_images.npz`

TODO: https://github.com/dchaley/deepcell-imaging/issues/25

