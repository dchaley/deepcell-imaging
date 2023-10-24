# DeepCell Imaging Notebooks

This is a set of notebooks based on the [deepcell sample notebook](https://github.com/vanvalenlab/deepcell-tf/blob/3234a52eb48b53f704590c1b649b5c8b19804a06/notebooks/applications/Mesmer-Application.ipynb). Each step is extracted into its own notebook, with the outputs persisted at each step.

This allows easy re-runs of any step, assuming the previous outputs remain unchanged.

This pipeline starts from a numpy array of numbers; the input TIFF image has already been processed to an extent.

- [Extract input channels](Extract-Channels.ipynb): extract nucleus + membrane channels from the image into numpy arrays
- [Visualize input](Visualize-Inputs.ipynb): combine channels into a single array + visualize with green (nucleus) & blue (membrane).
- [Predict segments from input channels](Predict-Segments.ipynb): run the model on the input channels to generate a segmentation
- [Visualize segments](Visualize-Segmentation.ipynb): overlay segments on the input RGB and generate output images

Some potential next steps:

- actually implement channel extraction from TIF files (see also: ark-analysis examples)
- with non-trivial TIF input, size the steps (with focus on prediction)
- parameterize paths properly, unify var names etc
- create an end-to-end pipeline notebook (?)
- consider automation? ex: upload npz, triggers cloud function to start visualization job

.. and make a demo of it all!

# How to run locally

This assumes unixen, if you're on Windows use `venv\Scripts\activate` instead.

```bash
# Create & activate a venv.
# Important: DeepCell only supports up to Python 3.10, not 3.11, as of 2023-07-08
python3.10 -m venv venv
source venv/bin/activate

# Install dependencies.
python -m pip install -r requirements.txt

# Install ipython
python -m pip install ipython
# Reload venv to refresh path.
source venv/bin/activate

# Open jupyter-lab
venv/bin/jupyter-lab
```

## Local benchmarking

### Without tensorflow-metal

Parallel cpu --> lots of "time" !
```
PID    COMMAND      %CPU      TIME     #TH    #WQ  #PORT MEM
84741  Python       765.0     66:54.67 65/13  1    88    8660M+

PID    COMMAND      %CPU      TIME     #TH    #WQ   #PORT MEM
95657  Python       95.0      02:03:16 51/1   1     74    14G
```

Peaks at 18gb.

deepcell debug logs:

```
6:45:54 PM
INFO:root:Converting image dtype to float
6:46:44 PM
DEBUG:Mesmer:Pre-processed data with mesmer_preprocess in 56.7202 s
6:46:46 PM
DEBUG:Mesmer:_tile_input finished in 1.9673 s
7:05:33 PM
DEBUG:Mesmer:_batch_predict finished in 1127.0536 s
7:06:39 PM
DEBUG:Mesmer:_untile_output finished in 65.9215 s
7:06:40 PM
DEBUG:Mesmer:Run model finished in 1251.8428 s
DEBUG:Mesmer:Post-processing results with mesmer_postprocess and kwargs: {'whole_cell_kwargs': {'maxima_threshold': 0.075, 'maxima_smooth': 0, 'interior_threshold': 0.2, 'interior_smooth': 2, 'small_objects_threshold': 15, 'fill_holes_threshold': 15, 'radius': 2}, 'nuclear_kwargs': {'maxima_threshold': 0.1, 'maxima_smooth': 0, 'interior_threshold': 0.2, 'interior_smooth': 2, 'small_objects_threshold': 15, 'fill_holes_threshold': 15, 'radius': 2}, 'compartment': 'whole-cell'}
/Users/davidhaley/dev/deepcell-imaging/notebooks/venv/lib/python3.10/site-packages/deepcell_toolbox/deep_watershed.py:108: UserWarning: h_maxima peak finding algorithm was selected, but the provided image is larger than 5k x 5k pixels.This will lead to slow prediction performance.
  warnings.warn('h_maxima peak finding algorithm was selected, '
7:11:32 PM
DEBUG:Mesmer:Post-processed results with mesmer_postprocess in 291.9621 s
DEBUG:Mesmer:_resize_output finished in 0.0 s
7:11:52 PM
Prediction finished in 1564.53423699399 s
```


### No tensorflow-metal

I wasn't able to test this, because DeepCell uses TF 2.8 but the TF 2.8 Metal/MacOS distro isn't available for 2.8 on MacOS 12 or 13.

https://developer.apple.com/metal/tensorflow-plugin/

```
python -m pip install tensorflow-metal
```


# How to run in Vertex AI

Hopefully just upload the notebooks to a managed notebook instance! 🤞🏻

NOTE from Lynn: Vertex AI includes a dedicated model training service (SaaS, pay by the minute), this is commonly used when 'productizing ML' processes.

# Questions

Does Predict process multiple inputs faster in parallel, or called in sequence? (Can we use less mem, if in sequence & same speed?)

