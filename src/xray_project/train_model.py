import argparse
from pathlib import Path

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, random_split
from torchvision import transforms

from xray_project.dataset.xray_dataset import ChestXrayDataset
from xray_project.models import MultiTaskXRayModel


def train_one_epoch(model, loader, optimizer, disease_loss_fn, gender_loss_fn, device):
    model.train()
    total_loss = 0

    for images, disease_labels, gender_labels in loader:
        images = images.to(device)
        disease_labels = disease_labels.to(device)
        gender_labels = gender_labels.to(device)

        optimizer.zero_grad()

        disease_logits, gender_logits = model(images)

        disease_loss = disease_loss_fn(disease_logits, disease_labels)
        gender_loss = gender_loss_fn(gender_logits, gender_labels)

        loss = disease_loss + gender_loss
        loss.backward()
        optimizer.step()

        total_loss += loss.item()

    return total_loss / len(loader)


def evaluate(model, loader, disease_loss_fn, gender_loss_fn, device):
    model.eval()
    total_loss = 0
    correct_gender = 0
    total_gender = 0

    with torch.no_grad():
        for images, disease_labels, gender_labels in loader:
            images = images.to(device)
            disease_labels = disease_labels.to(device)
            gender_labels = gender_labels.to(device)

            disease_logits, gender_logits = model(images)

            disease_loss = disease_loss_fn(disease_logits, disease_labels)
            gender_loss = gender_loss_fn(gender_logits, gender_labels)

            loss = disease_loss + gender_loss
            total_loss += loss.item()

            gender_preds = torch.argmax(gender_logits, dim=1)
            correct_gender += (gender_preds == gender_labels).sum().item()
            total_gender += gender_labels.size(0)

    gender_acc = correct_gender / total_gender if total_gender > 0 else 0
    return total_loss / len(loader), gender_acc


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--csv_file", required=True)
    parser.add_argument("--image_dir", required=True)
    parser.add_argument("--epochs", type=int, default=2)
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--output", default="outputs/xray_multitask_model.pt")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
    ])

    dataset = ChestXrayDataset(
        csv_file=args.csv_file,
        image_dir=args.image_dir,
        transform=transform,
    )

    num_diseases = len(dataset.disease_columns)

    train_size = int(0.8 * len(dataset))
    val_size = len(dataset) - train_size

    train_dataset, val_dataset = random_split(dataset, [train_size, val_size])

    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False)

    model = MultiTaskXRayModel(num_diseases=num_diseases).to(device)

    disease_loss_fn = nn.BCEWithLogitsLoss()
    gender_loss_fn = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=args.lr)

    for epoch in range(args.epochs):
        train_loss = train_one_epoch(
            model, train_loader, optimizer, disease_loss_fn, gender_loss_fn, device
        )
        val_loss, gender_acc = evaluate(
            model, val_loader, disease_loss_fn, gender_loss_fn, device
        )

        print(
            f"Epoch {epoch+1}/{args.epochs} | "
            f"Train Loss: {train_loss:.4f} | "
            f"Val Loss: {val_loss:.4f} | "
            f"Gender Acc: {gender_acc:.4f}"
        )

    Path("outputs").mkdir(exist_ok=True)
    torch.save(model.state_dict(), args.output)
    print(f"Saved model to {args.output}")


if __name__ == "__main__":
    main()
