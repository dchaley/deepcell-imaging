from google.cloud import storage
from google.cloud.storage import Blob


def get_blobs(blob_uri, client=None):
    if client is None:
        client = storage.Client()

    blob = Blob.from_string(blob_uri, client=client)
    bucket = client.bucket(blob.bucket.name)
    return [x.name for x in bucket.list_blobs(prefix=f"{blob.name}")]
