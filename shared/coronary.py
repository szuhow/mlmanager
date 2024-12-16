import datetime
import os
import re
import time
import json
import copy
import glob
import torch
import argparse
import mlflow
from dotenv import load_dotenv
from itertools import product
import numpy as np
from tqdm import tqdm
from abc import ABC, abstractmethod
from PIL import Image
from collections import defaultdict
import torch.nn as nn
import torch.optim as optim
from torch.optim import lr_scheduler
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, models, utils
from torchvision.transforms.functional import to_tensor, to_pil_image
from torch.utils.tensorboard import SummaryWriter

if torch.cuda.is_available():
    device = torch.device("cuda")
    print(f'PyTorch is using GPU: {torch.cuda.get_device_name(0)}')
elif torch.backends.mps.is_available():
    device = torch.device("mps")
    print("PyTorch is using MPS (Metal Performance Shaders)")
else:
    device = torch.device("cpu")
    print("GPU not available, using CPU")

MLFLOW_BACKEND: str | None = None
MLFLOW_TRACKING: bool = False

load_dotenv()

if os.environ.get("MLFLOW_BACKEND") is not None:
    MLFLOW_BACKEND = os.environ.get("MLFLOW_BACKEND")
    MLFLOW_TRACKING = True


def natural_sort_key(s):
    """
    Function to generate a natural sort key.
    """
    return [int(text) if text.isdigit() else text.lower() for text in re.split('([0-9]+)', s)]


def double_conv(in_channels, out_channels):
    return nn.Sequential(
        nn.Conv2d(in_channels, out_channels, 3, padding=1),
        nn.ReLU(inplace=True),
        nn.Conv2d(out_channels, out_channels, 3, padding=1),
        nn.ReLU(inplace=True)
    )


class UNet(nn.Module):

    def __init__(self, n_class):
        super().__init__()

        self.dconv_down1 = double_conv(1, 64)
        self.dconv_down2 = double_conv(64, 128)
        self.dconv_down3 = double_conv(128, 256)
        self.dconv_down4 = double_conv(256, 512)

        self.maxpool = nn.MaxPool2d(2)
        self.upsample = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)

        self.dconv_up3 = double_conv(256 + 512, 256)
        self.dconv_up2 = double_conv(128 + 256, 128)
        self.dconv_up1 = double_conv(128 + 64, 64)

        self.conv_last = nn.Conv2d(64, n_class, 1)


    def forward(self, x):
        conv1 = self.dconv_down1(x)
        x = self.maxpool(conv1)

        conv2 = self.dconv_down2(x)
        x = self.maxpool(conv2)

        conv3 = self.dconv_down3(x)
        x = self.maxpool(conv3)

        x = self.dconv_down4(x)

        x = self.upsample(x)
        x = torch.cat([x, conv3], dim=1)

        x = self.dconv_up3(x)
        x = self.upsample(x)
        x = torch.cat([x, conv2], dim=1)

        x = self.dconv_up2(x)
        x = self.upsample(x)
        x = torch.cat([x, conv1], dim=1)

        x = self.dconv_up1(x)

        out = self.conv_last(x)

        return out


class CoronaryDataset(Dataset):
    def __init__(self, image_paths, target_paths, transform, train=True):
        self.image_paths = image_paths
        self.target_paths = target_paths
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


class LossFunction(ABC):
    @abstractmethod
    def compute(self, pred, target):
        pass

class Dice(LossFunction):
    def compute(self, pred, target, smooth=1e-5):
        pred = pred.contiguous()
        target = target.contiguous()

        intersection = (pred * target).sum(dim=2).sum(dim=2)

        loss = (1 - ((2. * intersection + smooth) / (pred.sum(dim=2).sum(dim=2) + target.sum(dim=2).sum(dim=2) + smooth)))
        print(f'Loss dice: {loss.mean()}')
        return loss.mean()

class IoU(LossFunction):
    def compute(self, pred_soft, target, smooth=1e-5):
        intersection = (pred_soft * target).sum()
        union = pred_soft.sum() + target.sum() - intersection
        
        iou = (intersection + smooth) / (union + smooth)
        loss = 1 - iou
        
        print(f'Loss iou: {loss.item()}')
        return loss
    
class DebugDice(LossFunction):
    def compute(self, pred, target, smooth=1e-5):
        # Print detailed information about predictions and targets
        print("Pred stats:")
        print(f"  Min: {pred.min().item():.4f}")
        print(f"  Max: {pred.max().item():.4f}")
        print(f"  Mean: {pred.mean().item():.4f}")
        
        print("\nTarget stats:")
        print(f"  Min: {target.min().item():.4f}")
        print(f"  Max: {target.max().item():.4f}")
        print(f"  Mean: {target.mean().item():.4f}")
        
        # Soft thresholding instead of hard binarization
        pred_soft = torch.sigmoid(pred)
        
        # Compute intersection and sums with support for arbitrary dimensions
        intersection = (pred_soft * target).sum(dim=tuple(range(1, pred.dim())))
        pred_sum = pred_soft.sum(dim=tuple(range(1, pred.dim())))
        target_sum = target.sum(dim=tuple(range(1, target.dim())))
        
        print("\nIntersection stats:")
        print(f"  Mean intersection: {intersection.mean().item():.4f}")
        print(f"  Pred sum: {pred_sum.mean().item():.4f}")
        print(f"  Target sum: {target_sum.mean().item():.4f}")
        
        # Compute Dice coefficient and loss
        dice_coeff = (2. * intersection + smooth) / (pred_sum + target_sum + smooth)
        loss = 1 - dice_coeff
        
        print("\nDice coefficient:", dice_coeff.mean().item())
        print("Dice loss:", loss.mean().item())
        
        return loss.mean()

# Funkcja do obliczania straty łączonej BCE i Dice
def calc_loss(pred, target, metrics, bce_weight=0.1, loss_function=Dice()):
    # Binary Cross-Entropy Loss
    bce = F.binary_cross_entropy_with_logits(pred, target)
    pred_soft = torch.sigmoid(pred)
    # Compute specified loss
    loss_value = loss_function.compute(pred_soft, target)
    loss_name = loss_function.__class__.__name__.lower()
    
    # Combine BCE and the computed loss
    loss = bce * bce_weight + loss_value * (1 - bce_weight)
    
    # Update metrics dictionary with batch-aggregated values
    with torch.no_grad():
        batch_size = target.size(0)
        metrics['bce'] += bce.item() * batch_size
        metrics[loss_name] += loss_value.item() * batch_size
        metrics['loss'] += loss.item() * batch_size
    
    return loss

# Funkcja do drukowania metryk
def print_metrics(metrics, epoch_samples, phase):
    outputs = [f"{k}: {metrics[k] / epoch_samples:.4f}" for k in metrics.keys()]
    print(f"{phase}: {', '.join(outputs)}")

# Funkcja treningowa
def train_model(model, dataloaders, optimizer, scheduler, device, writer, num_epochs, bce_weight, loss_type):
    best_model_wts = copy.deepcopy(model.state_dict())
    best_loss = float('inf')
    # Mapowanie typów strat na odpowiednie klasy
    loss_functions = {
        'dice': Dice(),
        'iou': IoU(),
        'debugdice': DebugDice()
    }

    # Wybierz odpowiednią funkcję straty na podstawie loss_type
    if loss_type not in loss_functions:
        raise ValueError(f"Unknown loss_type: {loss_type}")
    loss_function = loss_functions[loss_type]

    if MLFLOW_TRACKING:
        mlflow.autolog()

    for epoch in range(num_epochs):
        print(f'Epoch {epoch}/{num_epochs - 1}')
        print('-' * 10)
        since = time.time()

        for phase in ['train', 'val']:
            if phase == 'train':
                # scheduler.step()
                model.train()
                print("LR:", optimizer.param_groups[0]['lr'])
            else:
                model.eval()

            metrics = defaultdict(float)
            epoch_samples = 0

            with torch.set_grad_enabled(phase == 'train'):
                for i, (inputs, labels) in enumerate(tqdm(dataloaders[phase], desc=f'{phase} Epoch {epoch}', position=0, leave=True)):
                    inputs, labels = inputs.to(device), labels.to(device)
                    optimizer.zero_grad()
                    outputs = model(inputs)
                    loss = calc_loss(outputs, labels, metrics, bce_weight, loss_function=loss_function)
                    if phase == 'train':
                        loss.backward()
                        optimizer.step()
                        
                    epoch_samples += inputs.size(0)

                    writer.add_scalar('Batch loss/{}'.format(phase), loss.item(), epoch * len(dataloaders[phase]) + i)
                    writer.add_scalar('Batch BCE/{}'.format(phase), metrics['bce'] / (i + 1), epoch * len(dataloaders[phase]) + i)
                    writer.add_scalar('Batch {}/{}'.format(loss_type, phase), metrics[loss_type] / (i + 1), epoch * len(dataloaders[phase]) + i)

            print_metrics(metrics, epoch_samples, phase)
            epoch_loss = metrics['loss'] / epoch_samples
            epoch_custom = metrics[loss_type] / epoch_samples
            epoch_bce = metrics['bce'] / epoch_samples
            writer.add_scalar('Epoch loss/{}'.format(phase), epoch_loss, epoch)
            writer.add_scalar('Epoch BCE/{}'.format(phase), epoch_bce, epoch)
            writer.add_scalar('Epoch {}/{}'.format(loss_type, phase), epoch_custom, epoch)
            
            if phase == 'val' and epoch_loss < best_loss:
                print("Saving best model from epoch:", epoch)
                best_loss = epoch_loss
                best_model_wts = copy.deepcopy(model.state_dict())
                torch.save(model.state_dict(), 'best_model.pth')

            # for name, param in model.named_parameters():
            #     writer.add_histogram(name, param, epoch)
        scheduler.step()
        time_elapsed = time.time() - since
        print(f'{time_elapsed // 60:.0f}m {time_elapsed % 60:.0f}s')

    print(f'Best val loss: {best_loss:.4f}')
    model.load_state_dict(best_model_wts)
    writer.close()
    return model

# Funkcja do konfiguracji treningu
def setup_training(model, config, transform):
    path = config['path']
    lr = config['lr']
    batch_size = config['batch_size']
    shuffle = config['shuffle']
    epochs = config['epochs']
    optimizer_name = config['optimizer']
    bce_weight = config['bce_weight']
    transform = transform
    step_size = 7
    gamma = 0.1


    image_dir = sorted(glob.glob(os.path.join(path, 'JPEGImages/*')), key=natural_sort_key)
    mask_dir = sorted(glob.glob(os.path.join(path, 'SegmentationClassPNG/*')), key=natural_sort_key)
    len_data = len(image_dir)

    train_size = 0.8

    train_image_paths = image_dir[:int(len_data*train_size)]
    test_image_paths = image_dir[int(len_data*train_size):]

    train_mask_paths = mask_dir[:int(len_data*train_size)]
    test_mask_paths = mask_dir[int(len_data*train_size):]

    train_dataset = CoronaryDataset(train_image_paths, train_mask_paths, train=True, transform=transform)
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=0)

    test_dataset = CoronaryDataset(test_image_paths, test_mask_paths, train=False, transform=transform)
    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=0)


    dataloaders = {
        'train': train_loader,
        'val': test_loader
    }

    model = model.to(device)    

    if str(optimizer_name) == 'adam':
        optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=1e-5)
    elif str(optimizer_name) == 'sgd':
        optimizer = optim.SGD(model.parameters(), lr=lr, momentum=0.9, weight_decay=1e-5)
    else:
        raise ValueError(f"Unknown optimizer: {optimizer_name}")

    scheduler = lr_scheduler.StepLR(optimizer, step_size=step_size, gamma=gamma)
    writer = SummaryWriter(comment=f' LR_{lr}_optimizer_{optimizer_name}_batch_size_{batch_size}_gamma_{gamma}_step_size_{step_size}_epochs_{epochs}_bce_weight_{bce_weight}')  
    return model, optimizer, scheduler, device, writer, dataloaders

def create_config(path, lr, batch_size, shuffle, epochs, optimizer_name, bce_weight, loss_type):
    return {
        'path': path,
        'lr': lr,
        'batch_size': batch_size,
        'shuffle': shuffle,
        'epochs': epochs,
        'optimizer': optimizer_name,
        'bce_weight': bce_weight,
        'loss_type': loss_type
    }

def setup_mlflow():
    if not MLFLOW_TRACKING:
        return None
    mlflow.set_tracking_uri(uri=MLFLOW_BACKEND)
    mlflow.set_experiment(f"coronary-{datetime.datetime.now().strftime('%Y-%m-%d-%H%M')}")

def train_and_evaluate(config, transform):
    model = UNet(1)
    model, optimizer, scheduler, device, writer, dataloaders = setup_training(model, config, transform)
    _ = setup_mlflow()
    model = train_model(
        model=model,
        dataloaders=dataloaders,
        optimizer=optimizer,
        device=device,
        scheduler=scheduler,
        writer=writer,
        num_epochs=config['epochs'],
        bce_weight=config['bce_weight'],
        loss_type=config['loss_type']
    )
    val_loss = evaluate_model(model, dataloaders['val'], device)
    return model, val_loss

# Funkcja do tuningu modelu
def tune_model(parameters, transform):
    best_model = None
    best_loss = float('inf')

    for path, lr, batch_size, shuffle, epochs, optimizer_name, bce_weight, loss_type in product(*parameters.values()):
        config = create_config(path, lr, batch_size, shuffle, epochs, optimizer_name, bce_weight, loss_type)
        model, val_loss = train_and_evaluate(config, transform)

        if val_loss < best_loss:
            best_loss = val_loss
            best_model = copy.deepcopy(model)

    return best_model

# Funkcja do oceny modelu
def evaluate_model(model, dataloader, device):
    model.eval()
    metrics = defaultdict(float)
    epoch_samples = 0

    with torch.no_grad():
        for inputs, labels in dataloader:
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            loss = calc_loss(outputs, labels, metrics)
            epoch_samples += inputs.size(0)

    epoch_loss = metrics['loss'] / epoch_samples
    return epoch_loss


def parse_list_or_value(value):
    try:
        # Spróbuj przekonwertować wartość na JSON
        parsed_value = json.loads(value)
        # Jeśli wartość nie jest listą, zwróć ją jako listę
        if not isinstance(parsed_value, list):
            return [parsed_value]
        return parsed_value
    except json.JSONDecodeError:
        # Jeśli wartość nie jest poprawnym JSON, zwróć ją jako listę
        return [value]

def get_resize_transforms(resolution):
    return transforms.Resize(resolution)
        

def main():

    parser = argparse.ArgumentParser(description='Training script')
    parser.add_argument('--path', type=parse_list_or_value, help='Dataset path')
    parser.add_argument('--epochs', type=parse_list_or_value, help='Number of epochs (e.g., 10 or [10,20])')
    parser.add_argument('--shuffle', type=parse_list_or_value, help='Shuffle dataset')
    parser.add_argument('--lr', type=parse_list_or_value, help='Learning rate')
    parser.add_argument('--batch_size', type=parse_list_or_value, help='Batch size')
    parser.add_argument('--optimizer', type=parse_list_or_value, help='Optimizer (e.g., adam or sgd)')
    parser.add_argument('--bce_weight', type=parse_list_or_value, help='BCE weight')
    parser.add_argument('--loss_type', type=parse_list_or_value, help='Loss type (dice or iou)')
    parser.add_argument('--halfres', action='store_true', help='Use half resolution (256x256)')
    parser.add_argument('--fullres', action='store_true', help='Use full resolution (512x512)')
    parser.add_argument('--dryrun', action='store_true', help='Dry run (no training)')
    args = parser.parse_args()

    if args.halfres:
        resolution = (256, 256)
    elif args.fullres:
        resolution = (512, 512)
    else:
        raise ValueError("You must specify either --halfres or --fullres")
    transform = get_resize_transforms(resolution)
    parameters = {
        'path': args.path,
        'lr': args.lr,
        'batch_size': args.batch_size,
        'shuffle': args.shuffle,
        'epochs': args.epochs,
        'optimizer': args.optimizer,
        'bce_weight': args.bce_weight,
        'loss_type': args.loss_type, 
    }

    print(f"Training with parameters:\n")
    for path, lr, batch_size, shuffle, epochs, optimizer_name, bce_weight, loss_type in product(*parameters.values()): # TODO: consider using ray.tune
        print(f"Path: {path}, \nLR: {lr}, \nBatch size: {batch_size}, \nShuffle: {shuffle}, \nEpochs: {epochs}, \nOptimizer: {optimizer_name}, \nBCE weight: {bce_weight}, \nLoss type: {loss_type}, \nResolution: {resolution}")
        print("\n")

    if not args.dryrun:
        tune_model(parameters, transform)


if __name__ == '__main__':
    main()