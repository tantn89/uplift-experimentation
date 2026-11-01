"""Dataset loaders: download on demand, verify SHA-256, never commit raw data."""

from uplift_lab.data.cache import ChecksumError, RemoteFile, cache_dir, fetch, sha256_file
from uplift_lab.data.criteo import CRITEO, build_subsample, load_criteo_subsample
from uplift_lab.data.hillstrom import HILLSTROM, load_hillstrom, read_hillstrom_csv
from uplift_lab.data.sampling import stratified_indices

__all__ = [
    "CRITEO",
    "HILLSTROM",
    "ChecksumError",
    "RemoteFile",
    "build_subsample",
    "cache_dir",
    "fetch",
    "load_criteo_subsample",
    "load_hillstrom",
    "read_hillstrom_csv",
    "sha256_file",
    "stratified_indices",
]
