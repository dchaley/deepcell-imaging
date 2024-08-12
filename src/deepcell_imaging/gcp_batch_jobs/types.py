from typing import Optional

from pydantic import BaseModel, Field, ConfigDict

DEFAULT_BATCH_SIZE = 16


class PreprocessArgs(BaseModel):
    image_uri: str = Field(
        title="Image URI",
        description="URI to input image npz file, containing an array named 'input_channels' by default (see --image-array-name)",
    )
    image_array_name: str = Field(
        default="input_channels",
        title="Image Array Name",
        description="Name of array in input image npz file. Default/blank: input_channels",
    )
    image_mpp: Optional[float] = Field(
        default=None,
        title="Image Microns Per Pixel",
        description="Microns per pixel of input image. Default/blank: use the model's mpp value.",
    )
    output_uri: str = Field(
        title="Output URI",
        description="Where to write preprocessed input npz file containing an array named 'image'",
    )
    benchmark_output_uri: str = Field(
        default="",
        title="Benchmark Output URI",
        description="Where to write preprocessing benchmarking data. Default/blank: don't write benchmarking data.",
    )


class PredictArgs(BaseModel):
    model_config = ConfigDict(protected_namespaces=())

    image_uri: str = Field(
        title="Image URI",
        description="URI to preprocessed image npz file, containing an array named 'image'",
    )
    batch_size: int = Field(
        default=DEFAULT_BATCH_SIZE,
        title="Batch Size",
        description=f"Optional integer representing batch size to use for prediction. Default is {DEFAULT_BATCH_SIZE}.",
    )
    output_uri: str = Field(
        title="Output URI",
        description="Where to write model output npz file containing arr_0, arr_1, arr_2, arr_3",
    )
    benchmark_output_uri: str = Field(
        default="",
        title="Benchmark Output URI",
        description="Where to write prediction benchmarking data. Default/blank: don't write benchmarking data.",
    )
    model_path: str = Field(
        title="Model Path",
        description="Path to the model archive",
    )
    model_hash: str = Field(
        title="Model Hash",
        description="The hash of the model archive",
    )


class PostprocessArgs(BaseModel):
    raw_predictions_uri: str = Field(
        title="Raw Predictions URI",
        description="URI to model output npz file, containing 4 arrays: arr_0, arr_1, arr_2, arr_3",
    )
    input_rows: int = Field(
        title="Input Rows",
        description="Number of rows in the input image.",
    )
    input_cols: int = Field(
        title="Input Columns",
        description="Number of columns in the input image.",
    )
    compartment: str = Field(
        default="whole-cell",
        title="Compartment",
        description="Compartment to segment. One of 'whole-cell' (default) or 'nuclear' or 'both'.",
    )
    output_uri: str = Field(
        title="Output URI",
        description="URI to write postprocessed segment predictions npz file containing an array named 'image'.",
    )
    tiff_output_uri: str = Field(
        default="",
        title="TIFF Output URI",
        description="Where to write segment predictions TIFF file containing a segment number for each pixel. Default/blank: don't write TIFF file.",
    )
    benchmark_output_uri: str = Field(
        default="",
        title="Benchmark Output URI",
        description="Where to write postprocessing benchmarking data. Default/blank: don't write benchmarking data.",
    )


class VisualizeArgs(BaseModel):
    image_uri: str = Field(
        title="Image URI",
        description="URI to input image npz file, containing an array named 'input_channels' by default (see --image_array_name)",
    )
    image_array_name: str = Field(
        default="input_channels",
        title="Image Array Name",
        description="Name of array in input image npz file, default: input_channels",
    )
    predictions_uri: str = Field(
        title="Predictions URI",
        description="URI to image predictions npz file, containing an array named 'image'",
    )
    visualized_input_uri: str = Field(
        title="Visualized Input URI",
        description="Where to write visualized input png file.",
    )
    visualized_predictions_uri: str = Field(
        title="Visualized Predictions URI",
        description="Where to write visualized predictions png file.",
    )


class GatherBenchmarkArgs(BaseModel):
    preprocess_benchmarking_uri: str = Field(
        title="Preprocess benchmarking URI",
        description="URI to benchmarking data for the preprocessing step.",
    )
    prediction_benchmarking_uri: str = Field(
        title="Prediction benchmarking URI",
        description="URI to benchmarking data for the prediction step.",
    )
    postprocess_benchmarking_uri: str = Field(
        title="Postprocess benchmarking URI",
        description="URI to benchmarking data for the postprocessing step.",
    )
    bigquery_benchmarking_table: str = Field(
        default="",
        title="BigQuery benchmarking table",
        description="The fully qualified name (project.dataset.table) of the BigQuery table to write benchmarking data to. Default/blank: don't write to BigQuery.",
    )
