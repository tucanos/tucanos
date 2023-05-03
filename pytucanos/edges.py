import numpy as np

TRI2EDG = [
    [0, 1],
    [1, 2],
    [2, 0],
]

TET2EDG = [
    [0, 1],
    [1, 2],
    [2, 0],
    [0, 3],
    [1, 3],
    [2, 3],
]


def edges(els):

    assert els.ndim == 2
    if els.shape[1] == 2:
        edgs = els.copy()
    elif els.shape[1] == 3:
        edgs = np.vstack([els[:, e] for e in TRI2EDG])
    elif els.shape[1] == 4:
        edgs = np.vstack([els[:, e] for e in TET2EDG])

    edgs.sort(axis=1)
    return np.unique(edgs, axis=0)
