from dataset_loader import ParkinsonSpectrogramDataset

dataset = ParkinsonSpectrogramDataset("spectrograms")

print("Total samples:", len(dataset))
img, label = dataset[0]
print("Image shape:", img.shape)
print("Label:", label)
