import logging
import os

from google.cloud import storage
from google.cloud.storage import Blob


logger = logging.getLogger(__name__)


def gs_uri_to_basename(gs_uri):
    filename = gs_uri.split("/")[-1]
    return filename.split(".")[0]


def get_blob_filenames(uri_prefix, client=None):
    if client is None:
        client = storage.Client()

    root_blob = Blob.from_string(uri_prefix, client=client)
    bucket = client.bucket(root_blob.bucket.name)
    return set(
        [
            gs_uri_to_basename(x.name)
            for x in bucket.list_blobs(prefix=f"{root_blob.name}")
        ]
    )


def find_matching_npz(image_names, npz_root, npz_names):
    for image in image_names:
        has_npz = image in npz_names

        if not has_npz:
            logger.info(f"Skipping {image}: no NPZ found.")
            continue

        logger.info(f"Found matching npz for image: {image}")

        yield image, f"{npz_root}/{image}.npz"
