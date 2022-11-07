import numpy as np
import quivr.dsl as dsl

def fast(Q, n):
    identity_mtx = np.full(Q.shape, -np.inf)
    np.fill_diagonal(identity_mtx, np.inf)
    base = np.maximum(identity_mtx, Q) # Q + I
    result = power(base, n-1) # (Q + I)^(n-1)
    result = np.amax(np.minimum(Q[..., np.newaxis], result[np.newaxis, ...]), axis=1) # Q * (Q + I)^(n-1)
    return result

def power(M, n):
    if n == 1:
        return M
    elif n % 2 == 0:
        M_squared = np.amax(np.minimum(M[..., np.newaxis], M[np.newaxis, ...]), axis=1)
        return power(M_squared, n // 2)
    else:
        M_squared = np.amax(np.minimum(M[..., np.newaxis], M[np.newaxis, ...]), axis=1)
        arr = power(M_squared, n // 2)
        return np.amax(np.minimum(M[..., np.newaxis], arr[np.newaxis, ...]), axis=1)

def naive(Q, n):
    identity_mtx = np.full(Q.shape, -np.inf)
    np.fill_diagonal(identity_mtx, np.inf)

    base_arr = [Q]

    for _ in range(2, n + 1):
        Q_pow_k = np.amax(np.minimum(base_arr[-1][..., np.newaxis], Q[np.newaxis, ...]), axis=1)
        base_arr.append(Q_pow_k)

    return np.amax(np.stack(base_arr, axis=0), axis=0)

if __name__ == "__main__":
    # random numpy array
    for _ in range(100):
        Q = np.random.rand(30, 30)
        n = 30
        fast_result = fast(Q, n)
        naive_result = naive(Q, n)
        assert np.allclose(fast_result, naive_result)
