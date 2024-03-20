## TissueNet 1.1 selection of 5 images

We picked these image indexes arbitrarily from the full TissueNet train data. First, load the data:

```
tissuenet = TissueNet(version='1.1')
X_val, y_val, meta_val = tissuenet.load_data(split='val')
X_train, y_train, meta_train = tissuenet.load_data(split='train')
X_test, y_test, meta_test = tissuenet.load_data(split='test')
```

Then we selected these 5 indexes from `X_train` and `y_train`:

* 10, 27, 133, 48, 194

We used [this notebook](../../../../notebooks/Download-DeepCell-Data.ipynb).

The result is these files:

- `selected_training_data.npz` : contains two arrays: `X` input image, and `y` annotations (note the case).
- `image_x_input.png` : RGB visualization of each input
- `image_x_prediction_cell.png` : RGB visualization of whole-cell predictions overlaid on input
- `image_x_prediction_nucleus.png` : RGB visualization of nucleus predictions overlaid on input

Example images: Training index #133

![](image_2_input.png)

![](image_2_prediction_cell.png)

![](image_2_prediction_nucleus.png)