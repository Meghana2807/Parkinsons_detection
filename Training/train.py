import torch
import torch.nn as nn
from torch.utils.data import DataLoader, random_split

from dataset_loader import ParkinsonSpectrogramDataset
from model import CNNTransformer


# -----------------------------
# DEVICE
# -----------------------------
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# -----------------------------
# LOAD DATASET
# -----------------------------
dataset = ParkinsonSpectrogramDataset(root_dir="../spectrograms")

# Split dataset (80% train, 20% test)
train_size = int(0.8 * len(dataset))
test_size = len(dataset) - train_size

train_dataset, test_dataset = random_split(dataset, [train_size, test_size])

train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=16, shuffle=False)


print(f"Total samples: {len(dataset)}")
print(f"Training samples: {train_size}")
print(f"Testing samples: {test_size}")


# -----------------------------
# MODEL
# -----------------------------
model = CNNTransformer(num_classes=2)
model.to(device)


# -----------------------------
# LOSS & OPTIMIZER
# -----------------------------
criterion = nn.CrossEntropyLoss()

optimizer = torch.optim.Adam(model.parameters(), lr=0.0001)


# -----------------------------
# TRAINING
# -----------------------------
epochs = 20

print("Training started...\n")

for epoch in range(epochs):

    model.train()
    total_loss = 0
    correct = 0
    total = 0

    for images, labels in train_loader:

        images = images.to(device)
        labels = labels.to(device)

        outputs = model(images)

        loss = criterion(outputs, labels)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        total_loss += loss.item()

        _, predicted = torch.max(outputs, 1)

        correct += (predicted == labels).sum().item()
        total += labels.size(0)

    train_accuracy = (correct / total) * 100

    print(f"Epoch [{epoch+1}/{epochs}] Loss: {total_loss:.4f} Accuracy: {train_accuracy:.2f}%")


# -----------------------------
# SAVE MODEL
# -----------------------------
torch.save(model.state_dict(), "../models/transformer_model.pth")

print("\nModel saved successfully!")