import fnmatch
import os
import string
import random

random.seed(10)

def recursive_glob_full_path(rootdir='.', pattern='*'):
    """Search recursively for files matching a specified pattern.

    Adapted from http://stackoverflow.com/questions/2186525/use-a-glob-to-find-files-recursively-in-python
    """

    matches_full_paths = []
    ids = []
    for root, dirnames, filenames in os.walk(rootdir):
        for filename in fnmatch.filter(filenames, pattern):
            matches_full_paths.append(os.path.join(root, filename))
            basename, ext = os.path.splitext(filename)
            ids.append(basename)

    return matches_full_paths, ids

def get_all_folders(in_dir):
    names = [d for d in os.listdir(in_dir) if os.path.isdir(os.path.join(in_dir, d))]
    
    folder_paths = []
    for name in names:
        folder_paths.append(os.path.join(in_dir, name))
        
    return folder_paths, names