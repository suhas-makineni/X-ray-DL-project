import os
import pandas as pd
from PIL import Image
import torch
from torch.utils.data import Dataset


class ChestXrayDataset(Dataset):
    def __init__(self, csv_file, image_dir, transform=None):
        self.data = pd.read_csv(csv_file)
        self.image_dir = image_dir
        self.transform = transform

        self.disease_columns = [
            col for col in self.data.columns
            if col not in ["Image Index", "Patient Age", "Patient Gender"]
        ]

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        row = self.data.iloc[idx]

        image_path = os.path.join(self.image_dir, row["Image Index"])
        image = Image.open(image_path).convert("RGB")

        if self.transform:
            image = self.transform(image)

        disease_labels = torch.tensor(
            row[self.disease_columns].values.astype("float32")
        )

        gender_label = 1 if row["Patient Gender"] == "M" else 0
        metadata_label = torch.tensor(gender_label, dtype=torch.long)

        return image, disease_labels, metadata_label

