import numpy as np


def normalize(arr: np.ndarray, eps: float = 1e-8) -> np.ndarray:
    return (arr - arr.mean()) / (arr.std() + eps)


def padding(arr: np.ndarray, pad_value: float, max_length: int) -> np.ndarray:
    input_length = arr.shape[0]
    if input_length > max_length:
        arr = arr[:max_length]
    else:
        pad_length = max_length - input_length
        arr = np.pad(arr, ((0, pad_length), (0, 0)), mode='constant', constant_values=pad_value)

    return arr
