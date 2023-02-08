import json
import pickle
import numpy as np

from config import SAVE_DIR, DATA_DIR


def get_file_text(filepath):
    with open(filepath) as f:
        data = json.load(f)
    return data['text'], data['id']


def get_article(id: int):
    """Returns the text and title of an article."""
    filepath = DATA_DIR + f'{id}.json'
    with open(filepath) as f:
        data = json.load(f)

    return data


def save_pickle(things: dict):
    """Saves things using pickle."""
    for name, thing in things.items():
        with open(f'{SAVE_DIR}{name}.pickle', "wb") as f:
            pickle.dump(thing, f)


def save_svd(svd):
    """Saves a sparse matrix as a .npz file."""
    np.savez_compressed(
        f'{SAVE_DIR}svd.npz',
        u = svd[0],
        s = svd[1],
        vt = svd[2]
        )


def load(things: list):
    """Loads things from pickle files."""
    loaded = []
    for name in things:
        if name == "svd":
            svd = np.load(SAVE_DIR+ 'svd.npz')
            svd = svd['u'], svd['s'], svd['vt']
            loaded.append(svd)
        else:
            with open(f'{SAVE_DIR}{name}.pickle', "rb") as f:
                loaded.append(pickle.load(f))

    return loaded
