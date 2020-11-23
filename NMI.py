import numpy as np
import math


def nmi(a: np.ndarray, b: np.ndarray):
    total = len(a)
    a_ids = np.unique(a)
    b_ids = np.unique(b)
    mi = 0
    for i in a_ids:
        for j in b_ids:
            a_occur = np.where(a == i)[0]
            b_occur = np.where(b == j)[0]
            ab_occur = np.array([x for x in a_occur if x in b_occur])

            px = len(a_occur) / total
            py = len(b_occur) / total
            pxy = len(ab_occur) / total

            mi = mi + pxy * math.log2(pxy / (px * py) + eps)

    hx = 0
    for i in a_ids:
        a_occur_count = len(np.where(a == i)[0])
        hx = hx + (a_occur_count / total) * math.log2(a_occur_count / total + eps)

    hy = 0
    for j in b_ids:
        b_occur_count = len(np.where(b == j)[0])
        hy = hy + (b_occur_count / total) * math.log2(b_occur_count / total + eps)

    nmi_value = -2 * mi / (hx + hy)
    return nmi_value


eps = 2.2204e-16
if __name__ == "__main__":
    A = np.array([1, 1, 1, 1, 1, 1, 2, 2, 2, 2, 2, 2, 3, 3, 3, 3, 3], dtype=int)
    B = np.array([1, 2, 1, 1, 1, 1, 1, 2, 2, 2, 2, 3, 1, 1, 3, 3, 3], dtype=int)
    print(nmi(A, B))
    # 0.36456177185719024
