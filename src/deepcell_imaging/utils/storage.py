import os

from google.cloud import storage
from google.cloud.storage import Blob


def get_blob_names(uri_prefix, client=None):
    if client is None:
        client = storage.Client()

    root_blob = Blob.from_string(uri_prefix, client=client)
    bucket = client.bucket(root_blob.bucket.name)
    return [x.name for x in bucket.list_blobs(prefix=f"{root_blob.name}")]


def find_matching_npz(image_names, npz_root, npz_names, client=None):
    if client is None:
        client = storage.Client()

    for image in image_names:
        image_name = os.path.splitext(os.path.basename(image))[0]
        npz_path = f"{npz_root}/{image_name}.npz"
        npz_path_blob = Blob.from_string(npz_path, client=client)
        has_npz = npz_path_blob.name in npz_names

        if not has_npz:
            continue

        yield image_name, npz_path
