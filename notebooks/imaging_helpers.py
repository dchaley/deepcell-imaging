import timeit

from deepcell.applications import Mesmer
import tensorflow as tf


def make_app(model_path):
    t = timeit.default_timer()
    model = tf.keras.models.load_model(model_path)
    print("Loaded model in %s s" % (timeit.default_timer() - t))

    return Mesmer(model=model)
