from deepcell_imaging.utils.storage import find_matching_npz


def test_find_matching_npz():
    image_names = ["image1", "image2", "image3"]
    npz_root = "npz_root"
    npz_names = ["image1", "image2"]

    result = list(find_matching_npz(image_names, npz_root, npz_names))

    assert result == [
        ("image1", "npz_root/image1.npz"),
        ("image2", "npz_root/image2.npz"),
    ]
