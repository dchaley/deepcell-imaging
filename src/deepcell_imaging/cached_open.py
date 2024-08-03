import hashlib
import logging
import os
import shutil
import smart_open
import tarfile
import urllib
import warnings
import zipfile

from urllib.parse import urlsplit


# NOTE: This code was copied wholesale from Keras:
# https://github.com/keras-team/keras/blob/v2.14.0/keras/utils/data_utils.py
# https://github.com/keras-team/keras/blob/v2.14.0/keras/utils/io_utils.py
# licensed under the Apache License 2
# https://github.com/keras-team/keras/blob/master/LICENSE
#
# It was then modified:
# - use smart_open reading in chunks, instead of urllib
# - remove the progress bar (unsupported by smart_open)
# - remove the deprecated `untar` & `md5_hash` parameters
#
# The `get_file` function is the interesting one, the rest are helpers.
#
# Caution: the urllib error handling hasn't been updated.
# Need to discover & fix equivalent smart_open errors.
#
# Potential next steps: refactor to a library outside deepcell-imaging
# Somebody else asked for such a library:
# https://www.reddit.com/r/learnpython/comments/pse41r/library_to_download_file_if_not_exists/


def _resolve_path(path):
    return os.path.realpath(os.path.abspath(path))


def _is_path_in_dir(path, base_dir):
    return _resolve_path(os.path.join(base_dir, path)).startswith(base_dir)


def _is_link_in_dir(info, base):
    tip = _resolve_path(os.path.join(base, os.path.dirname(info.name)))
    return _is_path_in_dir(info.linkname, base_dir=tip)


def _filter_safe_paths(members):
    base_dir = _resolve_path(".")
    for finfo in members:
        valid_path = False
        if _is_path_in_dir(finfo.name, base_dir):
            valid_path = True
            yield finfo
        elif finfo.issym() or finfo.islnk():
            if _is_link_in_dir(finfo, base_dir):
                valid_path = True
                yield finfo
        if not valid_path:
            warnings.warn(
                "Skipping invalid path during archive extraction: " f"'{finfo.name}'."
            )


def _extract_archive(file_path, path=".", archive_format="auto"):
    """Extracts an archive if it matches tar, tar.gz, tar.bz, or zip formats.

    Args:
        file_path: Path to the archive file.
        path: Where to extract the archive file.
        archive_format: Archive format to try for extracting the file.
            Options are `'auto'`, `'tar'`, `'zip'`, and `None`.
            `'tar'` includes tar, tar.gz, and tar.bz files.
            The default 'auto' is `['tar', 'zip']`.
            `None` or an empty list will return no matches found.

    Returns:
        True if a match was found and an archive extraction was completed,
        False otherwise.
    """
    if archive_format is None:
        return False
    if archive_format == "auto":
        archive_format = ["tar", "zip"]
    if isinstance(archive_format, str):
        archive_format = [archive_format]

    file_path = path_to_string(file_path)
    path = path_to_string(path)

    for archive_type in archive_format:
        if archive_type == "tar":
            open_fn = tarfile.open
            is_match_fn = tarfile.is_tarfile
        if archive_type == "zip":
            open_fn = zipfile.ZipFile
            is_match_fn = zipfile.is_zipfile

        if is_match_fn(file_path):
            with open_fn(file_path) as archive:
                try:
                    if zipfile.is_zipfile(file_path):
                        # Zip archive.
                        archive.extractall(path)
                    else:
                        # Tar archive, perhaps unsafe. Filter paths.
                        archive.extractall(path, members=_filter_safe_paths(archive))
                except (tarfile.TarError, RuntimeError, KeyboardInterrupt):
                    if os.path.exists(path):
                        if os.path.isfile(path):
                            os.remove(path)
                        else:
                            shutil.rmtree(path)
                    raise
            return True
    return False


def path_to_string(path):
    """Convert `PathLike` objects to their string representation.

    If given a non-string typed path object, converts it to its string
    representation.

    If the object passed to `path` is not among the above, then it is
    returned unchanged. This allows e.g. passthrough of file objects
    through this function.

    Args:
        path: `PathLike` object that represents a path

    Returns:
        A string representation of the path argument, if Python support exists.
    """
    if isinstance(path, os.PathLike):
        return os.fspath(path)
    return path


def _resolve_hasher(algorithm, file_hash=None):
    """Returns hash algorithm as hashlib function."""
    if algorithm == "sha256":
        return hashlib.sha256()

    if algorithm == "auto" and file_hash is not None and len(file_hash) == 64:
        return hashlib.sha256()

    # This is used only for legacy purposes.
    return hashlib.md5()


def _hash_file(fpath, algorithm="sha256", chunk_size=65535):
    """Calculates a file sha256 or md5 hash.

    Example:

    ```python
    _hash_file('/path/to/file.zip')
    'e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855'
    ```

    Args:
        fpath: Path to the file being validated.
        algorithm: Hash algorithm, one of `'auto'`, `'sha256'`, or `'md5'`.
            The default `'auto'` detects the hash algorithm in use.
        chunk_size: Bytes to read at a time, important for large files.

    Returns:
        The file hash.
    """
    if isinstance(algorithm, str):
        hasher = _resolve_hasher(algorithm)
    else:
        hasher = algorithm

    with open(fpath, "rb") as fpath_file:
        for chunk in iter(lambda: fpath_file.read(chunk_size), b""):
            hasher.update(chunk)

    return hasher.hexdigest()


def validate_file(fpath, file_hash, algorithm="auto", chunk_size=65535):
    """Validates a file against a sha256 or md5 hash.

    Args:
        fpath: path to the file being validated
        file_hash:  The expected hash string of the file.
            The sha256 and md5 hash algorithms are both supported.
        algorithm: Hash algorithm, one of 'auto', 'sha256', or 'md5'.
            The default 'auto' detects the hash algorithm in use.
        chunk_size: Bytes to read at a time, important for large files.

    Returns:
        Whether the file is valid
    """
    hasher = _resolve_hasher(algorithm, file_hash)

    if str(_hash_file(fpath, hasher, chunk_size)) == str(file_hash):
        return True
    else:
        return False


def _makedirs_exist_ok(datadir):
    os.makedirs(datadir, exist_ok=True)


def get_file(
    fname=None,
    origin=None,
    file_hash=None,
    cache_subdir="datasets",
    hash_algorithm="auto",
    extract=False,
    archive_format="auto",
    cache_dir=None,
    chunk_size_bytes=10000000,
):
    """Downloads a file from a URL if it not already in the cache.

    By default the file at the url `origin` is downloaded to the
    cache_dir `~/.keras`, placed in the cache_subdir `datasets`,
    and given the filename `fname`. The final location of a file
    `example.txt` would therefore be `~/.keras/datasets/example.txt`.

    Files in tar, tar.gz, tar.bz, and zip formats can also be extracted.
    Passing a hash will verify the file after download. The command line
    programs `shasum` and `sha256sum` can compute the hash.

    Example:

    ```python
    path_to_downloaded_file = tf.keras.utils.get_file(
        origin="https://storage.googleapis.com/download.tensorflow.org/example_images/flower_photos.tgz",
        extract=True,
    )
    ```

    Args:
        fname: Name of the file. If an absolute path `/path/to/file.txt` is
            specified the file will be saved at that location. If `None`, the
            name of the file at `origin` will be used.
        origin: Original URL of the file.
        file_hash: The expected hash string of the file after download.
            The sha256 and md5 hash algorithms are both supported.
        cache_subdir: Subdirectory under the Keras cache dir where the file is
            saved. If an absolute path `/path/to/folder` is
            specified the file will be saved at that location.
        hash_algorithm: Select the hash algorithm to verify the file.
            options are `'md5'`, `'sha256'`, and `'auto'`.
            The default 'auto' detects the hash algorithm in use.
        extract: True tries extracting the file as an Archive, like tar or zip.
        archive_format: Archive format to try for extracting the file.
            Options are `'auto'`, `'tar'`, `'zip'`, and `None`.
            `'tar'` includes tar, tar.gz, and tar.bz files.
            The default `'auto'` corresponds to `['tar', 'zip']`.
            None or an empty list will return no matches found.
        cache_dir: Location to store cached files, when None it
            defaults to `~/.keras/`.
        chunk_size_bytes: Read from origin this many bytes at a time.

    Returns:
        Path to the downloaded file.

    ⚠️ **Warning on malicious downloads** ⚠️

    Downloading something from the Internet carries a risk.
    NEVER download a file/archive if you do not trust the source.
    We recommend that you specify the `file_hash` argument
    (if the hash of the source file is known) to make sure that the file you
    are getting is the one you expect.
    """
    if origin is None:
        raise ValueError(
            'Please specify the "origin" argument (URL of the file ' "to download)."
        )

    if cache_dir is None:
        cache_dir = os.path.join(os.path.expanduser("~"), ".keras")
    datadir_base = os.path.expanduser(cache_dir)
    if not os.access(datadir_base, os.W_OK):
        datadir_base = os.path.join("/tmp", ".keras")
    datadir = os.path.join(datadir_base, cache_subdir)
    _makedirs_exist_ok(datadir)

    fname = path_to_string(fname)
    if not fname:
        fname = os.path.basename(urlsplit(origin).path)
        if not fname:
            raise ValueError(
                "Can't parse the file name from the origin provided: "
                f"'{origin}'."
                "Please specify the `fname` as the input param."
            )

    fpath = os.path.join(datadir, fname)

    download = False
    if os.path.exists(fpath):
        # File found; verify integrity if a hash was provided.
        if file_hash is not None:
            if not validate_file(fpath, file_hash, algorithm=hash_algorithm):
                logging.info(
                    "A local file was found, but it seems to be "
                    f"incomplete or outdated because the {hash_algorithm} "
                    "file hash does not match the original value of "
                    f"{file_hash} "
                    "so we will re-download the data."
                )
                download = True
    else:
        download = True

    if download:
        logging.info(f"Downloading data from {origin}")

        error_msg = "URL fetch failure on {}: {} -- {}"
        try:
            try:
                with smart_open.open(origin, "rb", compression="disable") as fin:
                    with open(fpath, "wb") as fout:
                        bytes_read = 0
                        while True:
                            read_data = fin.read(chunk_size_bytes)
                            if len(read_data) == 0:
                                break
                            bytes_read += len(read_data)
                            fout.write(read_data)
            except urllib.error.HTTPError as e:
                raise Exception(error_msg.format(origin, e.code, e.msg))
            except urllib.error.URLError as e:
                raise Exception(error_msg.format(origin, e.errno, e.reason))
        except (Exception, KeyboardInterrupt):
            if os.path.exists(fpath):
                os.remove(fpath)
            raise

        # Validate download if succeeded and user provided an expected hash
        # Security conscious users would get the hash of the file from a
        # separate channel and pass it to this API to prevent MITM / corruption:
        if os.path.exists(fpath) and file_hash is not None:
            if not validate_file(fpath, file_hash, algorithm=hash_algorithm):
                raise ValueError(
                    "Incomplete or corrupted file detected. "
                    f"The {hash_algorithm} "
                    "file hash does not match the provided value "
                    f"of {file_hash}."
                )

        logging.info(f"{bytes_read} bytes downloaded")

    if extract:
        _extract_archive(fpath, datadir, archive_format)

    return fpath
