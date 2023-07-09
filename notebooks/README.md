# What is this

Starting point: [deepcell sample notebook](https://github.com/vanvalenlab/deepcell-tf/blob/3234a52eb48b53f704590c1b649b5c8b19804a06/notebooks/applications/Mesmer-Application.ipynb)

A collection of notebooks to replicate the steps(???) in a cellular image segmentation.

This pipeline starts from an image file (j/k it's a numpy array of numbers):

- Extract input channels (nucleus + membrane) from the image (j/k just load the `multiplex_tissue` dataset)
- Visualize input:
  - Combine channels into a single array + visualize with green (nucleus) & blue (membrane).
- Predict segments from input channels
- Visualize segmentation
  - Overlay segments on the input RGB and generate output images

Next steps:

- actually implement channel extraction from TIF files (see also: ark-analysis examples)
- with non-trivial TIF input, size the steps (with focus on prediction)
- parameterize paths properly, unify var names etc
- load input + model from supplied path not hardcoded S3
- create an end-to-end pipeline notebook (?)
- consider automation? ex: upload npz, triggers cloud function to start visualization job

Make a demo of it all!

# How to run locally

It's probably possible to create a conda environment from requirements.txt but I didn't figure out how! 😅

```
# Create & activate a venv.
# Important: DeepCell only supports up to Python 3.10, not 3.11, as of 2023-07-08
python3.10 -m venv venv
source venv/bin/activate

# Install dependencies.
python -m pip install -r requirements.txt

# Open jupyter-lab
venv/bin/jupyter-lab
```

# How to run in Vertex AI

Hopefully just upload the notebooks to a managed notebook instance! 🤞🏻

# Questions

Does Predict process multiple inputs faster in parallel, or called in sequence? (Can we use less mem, if in sequence & same speed?)

