import os
from PIL import Image
import torch
from torch.utils.data import Dataset, DataLoader, random_split, Subset
import numpy as np
from torchvision import transforms

class NormalTargetDataset(Dataset):
    def __init__(self, path_normal, path_target, image_size=(224, 224), augment=False):
        self.image_paths = []
        self.labels = []

        # target (label 1)
        for fname in os.listdir(path_target):
            if fname.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp')):
                self.image_paths.append(os.path.join(path_target, fname))
                self.labels.append(1)

        # normal (label 0)
        for fname in os.listdir(path_normal):
            if fname.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp')):
                self.image_paths.append(os.path.join(path_normal, fname))
                self.labels.append(0)

        # transform
        transform_list = [transforms.Resize(image_size)]

        if augment:
            transform_list += [
                transforms.RandomHorizontalFlip(p=0.2),
                transforms.RandomVerticalFlip(p=0.2),
                transforms.RandomApply([
                    transforms.RandomRotation(15)
                ], p=0.2)
            ]

        transform_list += [
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
        ]

        self.transform = transforms.Compose(transform_list)

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        img_path = self.image_paths[idx]
        image = Image.open(img_path).convert('RGB')
        image = self.transform(image)
        label = self.labels[idx]

        return image, label
    
class TargetOnlyDataset(Dataset):
    def __init__(self, path_target, image_size=(224, 224)):
        self.image_paths = []

        for fname in os.listdir(path_target):
            if fname.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp')):
                self.image_paths.append(os.path.join(path_target, fname))

        transform_list = [transforms.Resize(image_size)]
        transform_list += [
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
        ]

        self.transform = transforms.Compose(transform_list)

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        img_path = self.image_paths[idx]
        image = Image.open(img_path).convert('RGB')
        image = self.transform(image)
        label = 1

        return image, label

def create_dataloaders(path_normal, path_target, batch_size=32, image_size=(224, 224), augment=False, val_ratio=0.1):
    full_dataset = NormalTargetDataset(path_normal, path_target, image_size=image_size, augment=augment)
    target_datset = TargetOnlyDataset(path_target, image_size=image_size)

    total_size = len(full_dataset)
    test_size = int(total_size * 0.1)
    train_size = total_size - test_size

    train_dataset, test_dataset = random_split(full_dataset, [train_size, test_size])

    val_size = int(train_size * val_ratio)
    val_indices = torch.randperm(train_size)[:val_size]
    val_dataset = Subset(train_dataset, val_indices)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    return target_datset, train_loader, val_loader, test_loader