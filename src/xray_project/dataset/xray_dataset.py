import os
import pandas as pd
from PIL import Image
import torch
from torch.utils.data import Dataset


NIH_DISEASES = [
    "Atelectasis",
    "Cardiomegaly",
    "Effusion",
    "Infiltration",
    "Mass",
    "Nodule",
    "Pneumonia",
    "Pneumothorax",
    "Consolidation",
    "Edema",
    "Emphysema",
    "Fibrosis",
    "Pleural_Thickening",
    "Hernia",
]


class ChestXrayDataset(Dataset):
    def __init__(self, csv_file, image_dir, transform=None):
        self.data = pd.read_csv(csv_file)
        self.image_dir = image_dir
        self.transform = transform
        self.disease_columns = NIH_DISEASES

        available_images = set(os.listdir(image_dir))
        self.data = self.data[self.data["Image Index"].isin(available_images)]
        self.data = self.data.reset_index(drop=True)

        if len(self.data) == 0:
            raise RuntimeError("No matching images found.")

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        row = self.data.iloc[idx]

        image_path = os.path.join(self.image_dir, row["Image Index"])
        image = Image.open(image_path).convert("RGB")

        if self.transform:
            image = self.transform(image)

        findings = str(row["Finding Labels"]).split("|")

        disease_labels = torch.tensor(
            [1.0 if disease in findings else 0.0 for disease in self.disease_columns],
            dtype=torch.float32,
        )

        gender_label = 1 if row["Patient Gender"] == "M" else 0
        metadata_label = torch.tensor(gender_label, dtype=torch.long)

        return image, disease_labels, metadata_label
