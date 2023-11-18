# DeepCell sample data

Sample data sourced from the Van Valen Lab's `deepcell-tf` [repository](https://github.com/vanvalenlab/deepcell-tf/).

## mesmer-sample-X

This data was created with the [Extract-Mesmer-Sample-Channels](../../notebooks/Extract-Mesmer-Sample-Channels.ipynb) notebook. It uses a dataset from [a previous commit](https://github.com/vanvalenlab/deepcell-tf/blob/9e5fe9ab6237b7529dd6603038e002121fea291a/deepcell/datasets/__init__.py) that is no longer available in the deepcell-tf main branch. It has, seemingly, been replaced by the "tissue net" samples (TODO: [confirm this](https://github.com/dchaley/deepcell-imaging/issues/33)).

Each sample directory contains the following files.

1. `input.png`

A green/blue image showing the nuclear & membrane channels respectively.

![mesmer-sample-3 input](https://github.com/dchaley/deepcell-imaging/blob/main/sample-data/deepcell/mesmer-sample-3/input.png)

2. `input_channels.npz`

An npz archive containing the array `input_channels`.

```
In : np.load('sample-data/deepcell/mesmer-sample-3/input_channels.npz')['input_channels'].shape
Out: (512, 512, 2)
```

The dimensions are: rows, columns, channels. There are two channels: nuclear & membrane, in that order.

3. `input_rgb.npz`

An npz archive containing the array `rgb`: pixels.

```
In : np.load('sample-data/deepcell/mesmer-sample-3/input_rgb.npz')['rgb'].shape
Out: (512, 512, 3)
```

The dimensions are: rows, columns, R/G/B.

4. `predicted_segments.png`

The input image, with segment predictions overlaid.

![mesmer-sample-3 predicted_segments](https://github.com/dchaley/deepcell-imaging/blob/main/sample-data/deepcell/mesmer-sample-3/predicted_segments.png)

5. `predictions.npz`

[TODO](https://github.com/dchaley/deepcell-imaging/issues/25)

6. `raw_output_image.npz`

[TODO](https://github.com/dchaley/deepcell-imaging/issues/25)

7. `raw_output_images.npz`

[TODO](https://github.com/dchaley/deepcell-imaging/issues/25)

