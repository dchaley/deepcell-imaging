from deepcell.applications import Application, Mesmer
from deepcell.applications.mesmer import mesmer_preprocess, format_output_mesmer


def noop(model_output, *args, **kwargs):
    return model_output["whole-cell"][-1]


class MesmerNoPostprocess(Mesmer):
    def __init__(self, model):
        if model is None:
            raise ValueError("Require a model")

        # NOTE: These need to be kept in sync with
        # the Mesmer application parameters
        super(Mesmer, self).__init__(
            model,
            model_image_shape=model.input_shape[1:],
            model_mpp=0.5,
            preprocessing_fn=mesmer_preprocess,
            postprocessing_fn=noop,
            format_model_output_fn=format_output_mesmer,
            dataset_metadata=Mesmer.dataset_metadata,
            model_metadata=Mesmer.model_metadata,
        )
