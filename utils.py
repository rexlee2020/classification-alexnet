from torch.utils.data import Dataset
import os
import pickle
import numpy as np
from PIL import Image

class CIFAR10Dataset(Dataset):
    def __init__(self, root_dir, transform=None, train=True):
        self.root_dir = root_dir
        self.transform = transform
        self.data = []
        self.targets = []

        if train:
            # Load the training dataset files
            for i in range(1, 6):
                file_path = os.path.join(self.root_dir, f"data_batch_{i}")
                with open(file_path, "rb") as f:
                    batch = pickle.load(f, encoding="bytes")
                self.data.append(batch[b"data"])
                self.targets.extend(batch[b"labels"])
        else:
            # Load the validation dataset file
            file_path = os.path.join(self.root_dir, "test_batch")
            with open(file_path, "rb") as f:
                batch = pickle.load(f, encoding="bytes")
            self.data.append(batch[b"data"])
            self.targets.extend(batch[b"labels"])

        self.data = np.concatenate(self.data)
        self.data = self.data.reshape((-1, 3, 32, 32))
        self.data = self.data.transpose((0, 2, 3, 1))

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        img = Image.fromarray(self.data[index])
        target = self.targets[index]

        if self.transform is not None:
            img = self.transform(img)

        return img, target
