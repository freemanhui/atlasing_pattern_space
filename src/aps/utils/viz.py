import matplotlib.pyplot as plt
import numpy as np

def scatter_labels(coords: np.ndarray, labels, title: str, path: str=None):
    plt.figure(figsize=(6,6))
    plt.scatter(coords[:,0], coords[:,1], s=16)
    for (x,y), lab in zip(coords, labels):
        plt.text(float(x), float(y), lab, fontsize=8)
    plt.title(title)
    plt.xlabel('z1')
    plt.ylabel('z2')
    plt.tight_layout()
    if path:
        plt.savefig(path, dpi=150)
    else:
        plt.show()
