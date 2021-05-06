import numpy as np

from method import Gauss, Sweep


def print_SLAU(a, b):
    for i in range(a.shape[0]):
        line = ''
        for j in range(a.shape[1]):
            if j == 0:
                line += f"{a[i][j]}*x{j + 1} "
                continue
            if a[i][j] < 0:
                line += f"- {abs(a[i][j])}*x{j + 1} "
            else:
                line += f"+ {a[i][j]}*x{j + 1} "
        print(f"{line}= {b[i]}")


def main():
    a = np.array([
        [4, 1, 1, 2],
        [1, 3, 2, -1],
        [2, -1, 5, 3],
        [4, 5, 4, -4],
    ], dtype='float')

    b = np.array([2, 2, -1, 8], dtype='float')
    print_SLAU(a, b)
    g = Gauss(a, b)
    print(g.solution())

    a1 = np.array([
        [2, 2, 0, 0],
        [-1, 2, -0.5, 0],
        [0, 1, -3, -1],
        [0, 0, 1, 2],
    ], dtype='float')
    b1 = np.array([1, 0, 2, 2])
    print_SLAU(a1, b1)
    s = Sweep(a1, b1)
    print(s.solution())


if __name__ == '__main__':
    main()
