import numpy as np
import cupy as cp
import time

if __name__ == '__main__':

    N = 4096

    # N^2
    A = cp.random.rand(N, N).astype(cp.float32)

    # N^2
    B = cp.random.rand(N, N).astype(cp.float32)

    # N^2 ouput cells with 2N compute each
    st = time.monotonic()
    C = A @ B

    et = time.monotonic()
    s = et - st

    operations = N*N+(2*N)
    gflop = operations / 1e9

    print(f'GLOP/s: {gflop/s:.2f}')
