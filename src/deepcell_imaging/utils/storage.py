import os

from google.cloud import storage
from google.cloud.storage import Blob


def get_blobs(blob_uri, client=None):
    if client is None:
        client = storage.Client()

    blob = Blob.from_string(blob_uri, client=client)
    bucket = client.bucket(blob.bucket.name)
    return [x.name for x in bucket.list_blobs(prefix=f"{blob.name}")]


def find_matching_npz(image_blobs, npz_root, npz_blobs, client=None):
    if client is None:
        client = storage.Client()

    for image in image_blobs:
        image_name = os.path.splitext(os.path.basename(image))[0]
        npz_path = f"{npz_root}/{image_name}.npz"
        npz_path_blob = Blob.from_string(npz_path, client=client)
        has_npz = npz_path_blob.name in npz_blobs

        if not has_npz:
            continue

        yield image_name, npz_path
