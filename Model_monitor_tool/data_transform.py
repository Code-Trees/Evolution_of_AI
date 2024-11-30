import torch
import torchvision.transforms as transforms

def get_data_transform(augmentations):
    """Generate data transformations based on the selected augmentation strategies."""
    transform_list = [transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,))]

    # Check if the input is a string (for default case)
    if isinstance(augmentations, str):
        if augmentations == 'none':
            return transforms.Compose(transform_list)

    # If it's a list of augmentations, process each one
    for augmentation in augmentations:
        if augmentation['type'] == 'rotation':
            angle = augmentation['params'].get('angle', 10)
            transform_list.insert(0, transforms.RandomRotation(angle))
        elif augmentation['type'] == 'translation':
            x = augmentation['params'].get('x', 0.1)
            y = augmentation['params'].get('y', 0.1)
            transform_list.insert(0, transforms.RandomAffine(0, translate=(x, y)))
        elif augmentation['type'] == 'scaling':
            factor = augmentation['params'].get('factor', 0.8)
            transform_list.insert(0, transforms.RandomResizedCrop(size=(28, 28), scale=(factor, 1.0)))
        elif augmentation['type'] == 'flipping':
            transform_list.insert(0, transforms.RandomHorizontalFlip())

    return transforms.Compose(transform_list)

# Default transformation without augmentation
data_transform = get_data_transform('none')