import warnings
warnings.filterwarnings('ignore')
import torch
from torch.utils.data import Dataset,ConcatDataset
from torchvision import datasets,transforms
import gc
import os
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from PIL import Image

class ThinAffineTransform:
    def __call__(self, img):
        # Define affine transformation matrix
        angle = 0  # No rotation
        scale = 1.0  # Maintain original scale
        shear = (0, 30)  # Apply horizontal shear
        width_scale = 0.7  # Compress width (thinner appearance)

        # Apply affine transform
        width, height = img.size
        new_width = int(width * width_scale)
        img = img.resize((new_width, height), Image.BILINEAR)
        return img


class MNISTDataLoader:
    def __init__(self, batch_size=64, seed=43):
        self.cuda = torch.cuda.is_available()
        self.batch_size = batch_size
        torch.manual_seed(seed)
        
        # Calculate mean and std
        self.mean, self.std = self._calculate_stats()

        # Initialize transforms
        self._train_transforms = transforms.Compose([
            # ThinAffineTransform(),
            transforms.Resize((28, 28)),#interpolation=transforms.InterpolationMode.NEAREST),
            # transforms.ColorJitter(brightness=0.1, contrast=0.1, saturation=0.1, hue=0.1),
            transforms.RandomPerspective(distortion_scale=0.2, p=0.5, interpolation=transforms.InterpolationMode.NEAREST),
            transforms.RandomRotation(degrees=15, expand=False, center=None, fill=(self.mean) ),
            transforms.ToTensor(), 
            transforms.Normalize((self.mean,),
                                  (self.std,)),
        ])

        self.test_transforms = transforms.Compose([
            transforms.Resize((28, 28)),#interpolation=transforms.InterpolationMode.NEAREST),
            transforms.ToTensor(),
            transforms.Normalize((self.mean,),
                                  (self.std,))
        ])

        self.dataloader_args = dict(
            shuffle=True,
            batch_size=self.batch_size,
            num_workers=4,
            pin_memory=True
        ) if self.cuda else dict(shuffle=True, batch_size=64)

    def _calculate_stats(self):
        # Load entire dataset in one batch
        temp_data1 = datasets.MNIST(
            root='.data',
            train=True,
            download=True,
            transform=transforms.ToTensor()  # Only convert to tensor without normalization
        )
        # Load test dataset
        temp_data2 = datasets.MNIST(
            root='.data',
            train=False,
            download=True,
            transform=transforms.ToTensor()
        )
        combined_dataset = ConcatDataset([temp_data1, temp_data2])

        loader1 = torch.utils.data.DataLoader(combined_dataset, batch_size=len(combined_dataset), shuffle=False)
        data1 = next(iter(loader1))[0]

        # Calculate mean and std
        mean = data1.mean().item()
        std = data1.std().item()

        # Clean up
        del temp_data1, temp_data2, loader1, data1
        gc.collect()

        return mean, std

    def get_data_loaders(self):
        train_data = datasets.MNIST(
            root='.data',
            train=True,
            download=True,
            transform=self._train_transforms
        )

        test_data = datasets.MNIST(
            root='.data',
            train=False,
            download=True,
            transform=self.test_transforms,
        )
        
        train_loader = torch.utils.data.DataLoader(train_data, **self.dataloader_args)
        test_loader = torch.utils.data.DataLoader(test_data, **self.dataloader_args)
        
        return train_loader, test_loader

    @property
    def train_transforms(self):
        return self._train_transforms
    
    @train_transforms.setter
    def train_transforms(self, transforms):
        self._train_transforms = transforms

if __name__ == '__main__':
    data_loader = MNISTDataLoader()
    train_loader, test_loader = data_loader.get_data_loaders()
    data_loader.mean, data_loader.std
    print(len(train_loader.dataset), len(test_loader.dataset))
