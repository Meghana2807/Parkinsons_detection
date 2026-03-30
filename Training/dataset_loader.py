import os
from PIL import Image
from torch.utils.data import Dataset
import torchvision.transforms as transforms


class ParkinsonSpectrogramDataset(Dataset):
    def __init__(self, root_dir):
        self.root_dir = root_dir
        self.image_paths = []
        self.labels = []

        # Loop through folders (HC and PD)
        for label_name in os.listdir(root_dir):
            folder_path = os.path.join(root_dir, label_name)

            if not os.path.isdir(folder_path):
                continue

            for file in os.listdir(folder_path):
                if file.endswith(".png"):
                    self.image_paths.append(os.path.join(folder_path, file))

                    # Label encoding
                    if label_name == "HC":
                        self.labels.append(0)   # Healthy
                    else:
                        self.labels.append(1)   # Parkinson

        # Transformations
        self.transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.RandomHorizontalFlip(),
            transforms.RandomRotation(10),
            transforms.ToTensor()
        ])

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        image_path = self.image_paths[idx]
        label = self.labels[idx]

        try:
            image = Image.open(image_path).convert("RGB")
        except:
            # Skip corrupted image
            return self.__getitem__((idx + 1) % len(self.image_paths))

        image = self.transform(image)

        return image, label