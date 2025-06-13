from torch.utils.data import Dataset
from PIL import Image
from torchvision import transforms
import torch
import numpy as np

def is_image_file(filename):
    return any(str(filename).endswith(extension) for extension in [".jpg", ".jpeg", ".png", ".bmp", ".gif"])

class CoronaryDataset(Dataset):
    def __init__(self, image_paths, target_paths, transform, train=True):
        self.image_paths = [p for p in image_paths if is_image_file(p)]
        self.target_paths = [p for p in target_paths if is_image_file(p)]
        self.transforms = transforms.Compose([
            transforms.Grayscale(num_output_channels=3),  # Konwersja obrazu w skali szarości na obraz RGB
            transform,  # Resize 
            transforms.ToTensor(),
            # transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])  # wartości dla ImageNet
        ])
        self.mask_transforms = transforms.Compose([
            transform,  # Resize 
            transforms.ToTensor()
        ])

    def __getitem__(self, index):
        # Load image and mask
        image = Image.open(self.image_paths[index]).convert('RGB')  # Ensure RGB mode
        mask = Image.open(self.target_paths[index]).convert('RGB')  # Ensure RGB mode

        # Extract red channel from mask and copy to green and blue
        mask_np = np.array(mask)  # Convert to NumPy array
        red_channel = mask_np[:, :, 0]  # Extract red channel
        mask_np[:, :, 1] = red_channel  # Copy to green channel
        mask_np[:, :, 2] = red_channel  # Copy to blue channel

        # Convert back to PIL image and apply transformations
        updated_mask = Image.fromarray(mask_np)
        t_image = self.transforms(image)
        t_mask = self.mask_transforms(updated_mask)

        # Convert mask to single channel
        t_mask = t_mask[0, :, :].unsqueeze(0)  # Use only the red channel and add channel dimension

        # Binarize the mask if needed (e.g., threshold)
        t_mask = (t_mask > 0).type(torch.float32)
        grayscale = transforms.Grayscale(num_output_channels=1)

        t_image = grayscale(t_image)
        t_mask = grayscale(t_mask)
        return [t_image, t_mask]

    def __len__(self):
        return len(self.image_paths)
