import os

from boomlet.parallel import pmap
from boomlet.storage import joblib_dump


def _folder_autorun_helper(filename, func):
    result = func(filename)
    joblib_dump(filename + ".pkl", result)


def folder_apply(func, folder, ext, parallel=False):
    """
    Applies a function on the filenames of all files in a folder
    with a given extension (not ".pkl" or ".npy"), and serializes
    the output.
    """
    assert ext.startswith(".")
    assert ext not in (".pkl", ".npy")
    all_files = set(os.listdir(folder))
    ext_files = filter(lambda x: x.endswith(ext), all_files)
    new_files = filter(lambda x: (x + ".pkl") not in all_files, ext_files)
    new_file_paths = [os.path.join(folder, x) for x in new_files]
    if parallel:
        pmap(_folder_autorun_helper, new_file_paths, func)
    else:
        map(lambda x: _folder_autorun_helper(x, func), new_file_paths)
    return new_files
