import os
from PIL import Image
import matplotlib.pyplot as plt
from torch.utils.data import Dataset

def pil_collate_fn(batch):
        images = [item[0] for item in batch]
        labels = [item[1] for item in batch]
        return images, labels

class Dataset(Dataset):
    def __init__(self, patches_path, labels_file, transforms=None):
        self.patches_path = patches_path # root dir of dataset
        self.img_paths = [] # list of imagens paths
        self.img_labels = [] # list of imagens labels

        with open(labels_file) as file:
            crude_lines = file.readlines() # read all lines and return a list of strings

        for line in crude_lines:
            clean_line = line.split()
            self.img_paths.append(os.path.join(self.patches_path, clean_line[0]))
            self.img_labels.append(clean_line[1])
        
        self.transforms = transforms

    def __len__(self):
        return len(self.img_labels)

    def __getitem__(self, idx):
        img_path = self.img_paths[idx]
        img = Image.open(img_path)
        label = self.img_labels[idx]
        if self.transforms is not None:
            img = self.transforms(img)
        return img, label