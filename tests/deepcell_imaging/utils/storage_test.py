from unittest.mock import patch, Mock

from deepcell_imaging.utils.storage import find_matching_npz, get_blob_filenames


def test_get_blob_filenames():
    uri_prefix = "gs://bucket/images/"
    client = None

    urls = [
        f"{uri_prefix}/{x}" for x in ["image1.tiff", "image2.tiff", "image3.tiff", ""]
    ]

    with patch("google.cloud.storage.Blob") as mock_blob:
        mock_blob.from_string.return_value.bucket.name = "bucket"
        mock_blob.from_string.return_value.name = "images"

        with patch("google.cloud.storage.Client") as mock_client:
            mocks = [Mock() for _ in urls]
            for mock, url in zip(mocks, urls):
                mock.name = url

            mock_client.return_value.bucket.return_value.list_blobs.return_value = mocks

            result = get_blob_filenames(uri_prefix, client)

    assert result == {"image1", "image2", "image3"}


def test_find_matching_npz():
    image_names = ["image1", "image2", "image3"]
    npz_root = "npz_root"
    npz_names = ["image1", "image2"]

    result = list(find_matching_npz(image_names, npz_root, npz_names))

    assert result == [
        ("image1", "npz_root/image1.npz"),
        ("image2", "npz_root/image2.npz"),
    ]
