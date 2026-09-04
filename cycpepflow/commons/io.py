import os
import pickle


def load_pkl(file_path: str):
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"File {file_path} does not exist.")
    with open(file_path, "rb") as f:
        return pickle.load(f)


def save_pkl(file_path: str, data):
    with open(file_path, "wb") as f:
        pickle.dump(data, f)
