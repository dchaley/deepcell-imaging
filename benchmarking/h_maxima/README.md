# h_maxima optimization

Context here: [h_maxima performance](https://github.com/dchaley/deepcell-imaging/issues/8)

TLDR, this work presents a 5-15x optimization for gray reconstruction. The larger the image the greater the speedup.

This directory is the WIP area for prepping the code for merge into scikit.

Apply the (adapted) scikit test suite to the optimized code:

```
pytest test_reconstruction.py
```

