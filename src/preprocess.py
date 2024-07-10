import os
import numpy as np
import torch
from tqdm import tqdm
from scipy.signal import windows
from sklearn.decomposition import FastICA


def main():
    data_dir = "data"

    dtypes = ["train", "val", "test"]
    X = {dtype: torch.load(os.path.join(data_dir, f"{dtype}_X.pt")) for dtype in dtypes}
    X = {dtype: train_X.to('cpu').detach().numpy().copy() for dtype, train_X in X.items()}

    num_samples, num_channels, time_length = X['train'].shape
    window_fnc = windows.hann(time_length)

    skip_num = 10
    data_list = [load_process_data(idx, X, window_fnc) for idx in range(0, num_samples, skip_num)]
    conected_data = np.concatenate(data_list, axis=1)

    ica = FastICA(n_components=100)
    ica.fit(conected_data.T)

    for dtype in dtypes:
        preprocessed_data_list = []
        for idx in tqdm(range(X[dtype].shape[0])):
            preprocessed_data = load_process_data(idx, X, window_fnc, dtype)
            preprocessed_data = ica.transform(preprocessed_data.T)
            preprocessed_data_list.append(np.array(preprocessed_data.T))
        preprocessed_data_list = np.array(preprocessed_data_list)
        torch.save(torch.tensor(preprocessed_data_list), os.path.join(data_dir, f"Preprocessed_{dtype}_X.pt"))


def load_process_data(idx, X, window_fnc, dtype="train"):
    target_data = X[dtype][idx, :, :]
    mean = np.mean(target_data, axis=1, keepdims=True)
    std = np.std(target_data, axis=1, keepdims=True)
    standardized_data = (target_data - mean) / std
    processed_data = standardized_data * window_fnc
    return processed_data


if __name__ == "__main__":
    main()
