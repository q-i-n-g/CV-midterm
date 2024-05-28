import pickle
from pathlib import Path

def unpickle(fname):
    file = open(fname,'rb')
    obj = pickle.load(file)
    file.close()
    return obj

def list_subdirectories(path):
    p = Path(path)
    return [x.name for x in p.iterdir() if x.is_dir()]