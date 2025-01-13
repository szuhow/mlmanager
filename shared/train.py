import copy
import time
from pathlib import Path
from typing import Dict, Tuple, Any
import torch
from torch import nn, optim
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
import mlflow
import mlflow.pytorch
from collections import defaultdict
from datetime import datetime
from tqdm import tqdm
import os
from dotenv import load_dotenv
import argparse
import itertools
from torchvision import transforms
import json
from unet.unet import UNet
from datasets.coronary_dataset import CoronaryDataset
from utils.config import Config
from utils.loss import calc_loss, Dice, IoU, DebugDice

class TrainingManager:
    def __init__(self, config: Config, transform=None):
        self.config = config
        self.transform = transform
        self.device = self._setup_device()
        self.model = self._setup_model()
        self.dataloaders = self._setup_dataloaders()
        self.optimizer = self._setup_optimizer()
        self.scheduler = self._setup_scheduler()
        self.writer = self._setup_tensorboard()
        self.loss_function = self._setup_loss_function()
        
    def _setup_device(self) -> torch.device:
        if torch.cuda.is_available():
            device = torch.device("cuda")
            print(f'Using GPU: {torch.cuda.get_device_name(0)}')
        elif torch.backends.mps.is_available():
            device = torch.device("mps")
            print("Using MPS")
        else:
            device = torch.device("cpu")
            print("Using CPU")
        return device
    
    def _setup_model(self) -> nn.Module:
        model = UNet(1).to(self.device)
        return model
    
    def _print_epoch_summary(self, epoch_start: float, epoch_loss: float, best_loss: float):
        epoch_time = time.time() - epoch_start
        print(f'Epoch completed in {epoch_time:.0f}s')
        print(f'Current loss: {epoch_loss:.4f}, Best loss so far: {best_loss:.4f}')


    def _setup_dataloaders(self) -> Dict[str, DataLoader]:
        train_dataset, val_dataset = self._split_dataset()
        return {
            'train': DataLoader(train_dataset, batch_size=self.config.batch_size, shuffle=True),
            'val': DataLoader(val_dataset, batch_size=self.config.batch_size, shuffle=False)
        }
    
    def _split_dataset(self) -> Tuple[CoronaryDataset, CoronaryDataset]:
        path = Path(self.config.path)
        images = sorted(list(path.glob('JPEGImages/*')))
        masks = sorted(list(path.glob('SegmentationClassPNG/*')))
        
        split_idx = int(len(images) * 0.8)
        train_images, val_images = images[:split_idx], images[split_idx:]
        train_masks, val_masks = masks[:split_idx], masks[split_idx:]
        
        train_dataset = CoronaryDataset(train_images, train_masks, train=True, transform=self.transform)
        val_dataset = CoronaryDataset(val_images, val_masks, train=False, transform=self.transform)
        
        return train_dataset, val_dataset
    
    def _setup_optimizer(self) -> optim.Optimizer:
        optimizers = {
            'adam': optim.Adam(self.model.parameters(), lr=self.config.lr, weight_decay=1e-5),
            'sgd': optim.SGD(self.model.parameters(), lr=self.config.lr, momentum=0.9, weight_decay=1e-5)
        }
        if self.config.optimizer_name not in optimizers:
            raise ValueError(f"Unsupported optimizer: {self.config.optimizer_name}")
        return optimizers[self.config.optimizer_name]
    
    def _setup_scheduler(self) -> optim.lr_scheduler.StepLR:
        return optim.lr_scheduler.StepLR(self.optimizer, step_size=7, gamma=0.1)
    
    def _setup_tensorboard(self) -> SummaryWriter:
        return SummaryWriter(comment=self._get_tensorboard_comment())
    
    def _get_tensorboard_comment(self) -> str:
        return (f" LR_{self.config.lr}_optimizer_{self.config.optimizer_name}"
                f"_batch_size_{self.config.batch_size}_epochs_{self.config.epochs}"
                f"_bce_weight_{self.config.bce_weight}")
    
    def _setup_loss_function(self):
        loss_functions = {
            'dice': Dice(),
            'iou': IoU(),
            'debugdice': DebugDice()
        }
        if self.config.loss_type not in loss_functions:
            raise ValueError(f"Unsupported loss type: {self.config.loss_type}")
        return loss_functions[self.config.loss_type]
    
    def train(self) -> nn.Module:
        best_model_wts = copy.deepcopy(self.model.state_dict())
        best_loss = float('inf')
        
        for epoch in range(self.config.epochs):
            print(f'Epoch {epoch}/{self.config.epochs - 1}')
            print('-' * 10)
            epoch_start = time.time()
            
            for phase in ['train', 'val']:
                metrics = self._run_epoch(phase, epoch)
                epoch_loss = self._log_metrics(metrics, phase, epoch)
                
                if phase == 'val':
                    self.scheduler.step(epoch_loss)
                    best_model_wts = self._save_if_best(epoch_loss, epoch, best_loss, best_model_wts)
                    best_loss = min(epoch_loss, best_loss)
            
            self._print_epoch_summary(epoch_start, epoch_loss, best_loss)
        
        self._save_final_model(best_model_wts, best_loss)
        return self.model
    
    def _run_epoch(self, phase: str, epoch: int) -> defaultdict:
        self.model.train(phase == 'train')
        metrics = defaultdict(float)
        
        with torch.set_grad_enabled(phase == 'train'):
            for i, (inputs, labels) in enumerate(tqdm(self.dataloaders[phase], 
                                                    desc=f'{phase} Epoch {epoch}', position=0, leave=True)):
                loss = self._process_batch(inputs, labels, metrics, phase)
                self._log_batch_metrics(metrics, phase, epoch, i, len(self.dataloaders[phase]))
        
        return metrics
    
    def _process_batch(self, inputs: torch.Tensor, labels: torch.Tensor, 
                      metrics: defaultdict, phase: str) -> torch.Tensor:
        inputs = inputs.to(self.device)
        labels = labels.to(self.device)
        
        if phase == 'train':
            self.optimizer.zero_grad()
            
        outputs = self.model(inputs)
        loss = calc_loss(outputs, labels, metrics, self.config.bce_weight, 
                        loss_function=self.loss_function)
        
        if phase == 'train':
            loss.backward()
            self.optimizer.step()
            
        return loss
    
    def _log_batch_metrics(self, metrics: defaultdict, phase: str, 
                          epoch: int, batch_idx: int, epoch_len: int):
        self.writer.add_scalar(f'Batch loss/{phase}', 
                             metrics['loss'] / (batch_idx + 1), 
                             epoch * epoch_len + batch_idx)
        self.writer.add_scalar(f'Batch BCE/{phase}', 
                             metrics['bce'] / (batch_idx + 1), 
                             epoch * epoch_len + batch_idx)
        self.writer.add_scalar(f'Batch {self.config.loss_type}/{phase}', 
                             metrics[self.config.loss_type] / (batch_idx + 1), 
                             epoch * epoch_len + batch_idx)
    
    def _log_metrics(self, metrics: defaultdict, phase: str, epoch: int) -> float:
        samples = len(self.dataloaders[phase].dataset)
        epoch_loss = metrics['loss'] / samples
        epoch_custom = metrics[self.config.loss_type] / samples
        epoch_bce = metrics['bce'] / samples
        
        self.writer.add_scalar(f'Epoch loss/{phase}', epoch_loss, epoch)
        self.writer.add_scalar(f'Epoch BCE/{phase}', epoch_bce, epoch)
        self.writer.add_scalar(f'Epoch {self.config.loss_type}/{phase}', epoch_custom, epoch)
        self.writer.add_scalar('Learning_rate', self.optimizer.param_groups[0]['lr'], epoch)
        
        mlflow.log_metric(key="epoch_loss", value=epoch_loss, step=epoch)
        mlflow.log_metric(key="epoch_bce", value=epoch_bce, step=epoch)
        
        return epoch_loss
    
    def _save_if_best(self, epoch_loss: float, epoch: int, 
                     best_loss: float, best_weights: Dict) -> Dict:
        if epoch_loss < best_loss:
            print(f"Saving best model from epoch {epoch} with loss: {epoch_loss:.4f}")
            best_weights = copy.deepcopy(self.model.state_dict())
            mlflow.log_metric(key="best_loss", value=epoch_loss, step=epoch)
            self._save_checkpoint(epoch, epoch_loss)
        return best_weights
    
    def _save_checkpoint(self, epoch: int, loss: float):
        save_path = f'models/best_model_epoch_{epoch}_loss_{loss:.4f}.pth'
        torch.save({
            'epoch': epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'loss': loss,
        }, save_path)
    
    def _save_final_model(self, best_weights: Dict, best_loss: float):
        self.model.load_state_dict(best_weights)
        save_path = f'final_model_epochs_{self.config.epochs}_loss_{best_loss:.4f}.pth'
        torch.save({
            'epochs_completed': self.config.epochs,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'final_loss': best_loss,
        }, save_path)
        
        timestamp = datetime.now().strftime('%Y-%m-%d-%H%M')
        mlflow.pytorch.log_model(self.model, f"unet-{timestamp}")
        self.writer.close()

# def parse_list_or_value(value: str) -> list | str:
#     """Parse a string into either a list or a single value."""
#     if value.startswith('[') and value.endswith(']'):
#         return [item.strip() for item in value[1:-1].split(',')]
#     return value

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
        

def parse_arguments():
    """Parse command line arguments with improved validation and help messages."""
    parser = argparse.ArgumentParser(
        description='Training script for coronary artery segmentation',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    
    # Required arguments
    parser.add_argument(
        '--path',
        type=parse_list_or_value,
        required=True,
        help='Dataset path(s). Can be single path or list: "/data/path" or "[/path1, /path2]"'
    )
    
    # Training parameters
    training_group = parser.add_argument_group('Training Parameters')
    training_group.add_argument(
        '--epochs',
        type=parse_list_or_value,
        default='10',
        help='Number of epochs. Single value or list: "10" or "[10, 20]"'
    )
    training_group.add_argument(
        '--lr',
        type=parse_list_or_value,
        default='0.001',
        help='Learning rate(s). Single value or list: "0.001" or "[0.001, 0.0001]"'
    )
    training_group.add_argument(
        '--batch_size',
        type=parse_list_or_value,
        default='4',
        help='Batch size(s). Single value or list: "4" or "[4, 8]"'
    )
    
    # Model configuration
    model_group = parser.add_argument_group('Model Configuration')
    model_group.add_argument(
        '--optimizer',
        type=parse_list_or_value,
        default='adam',
        choices=['adam', 'sgd', '[adam,sgd]'],
        help='Optimizer choice(s). Single value or list: "adam" or "[adam, sgd]"'
    )
    model_group.add_argument(
        '--loss_type',
        type=parse_list_or_value,
        default='dice',
        choices=['dice', 'iou', 'debugdice', '[dice,iou]'],
        help='Loss function type(s). Single value or list: "dice" or "[dice, iou]"'
    )
    model_group.add_argument(
        '--bce_weight',
        type=parse_list_or_value,
        default='0.5',
        help='BCE weight in loss function. Single value or list: "0.5" or "[0.3, 0.5]"'
    )
    
    # Data handling
    data_group = parser.add_argument_group('Data Handling')
    resolution_group = data_group.add_mutually_exclusive_group(required=True)
    resolution_group.add_argument(
        '--halfres',
        action='store_true',
        help='Use half resolution (256x256)'
    )
    resolution_group.add_argument(
        '--fullres',
        action='store_true',
        help='Use full resolution (512x512)'
    )
    data_group.add_argument(
        '--shuffle',
        type=parse_list_or_value,
        default='True',
        help='Whether to shuffle the dataset. Single value or list: "True" or "[True, False]"'
    )
    
    # Execution control
    parser.add_argument(
        '--dryrun',
        action='store_true',
        help='Perform a dry run (print parameters without training)'
    )
    
    args = parser.parse_args()
    
    # Validate arguments
    if args.dryrun:
        print("Dry run mode - parameters will be displayed without training")
    
    return args

def get_resolution(args):
    """Determine resolution based on command line arguments."""
    if args.halfres:
        return (256, 256)
    elif args.fullres:
        return (512, 512)
    else:
        raise ValueError("Must specify either --halfres or --fullres")

def get_training_parameters(args):
    """Convert parsed arguments into training parameters dictionary."""
    return {
        'path': args.path if isinstance(args.path, list) else [args.path],
        'lr': _ensure_list(args.lr),
        'batch_size': _ensure_list(args.batch_size),
        'shuffle': _ensure_list(args.shuffle),
        'epochs': _ensure_list(args.epochs),
        'optimizer': _ensure_list(args.optimizer),
        'bce_weight': _ensure_list(args.bce_weight),
        'loss_type': _ensure_list(args.loss_type)
    }

def _ensure_list(value):
    """Ensure value is a list."""
    if isinstance(value, list):
        return value
    return [value]

def run_training(parameters, resolution):
    """Run the training process with MLflow tracking."""
    mlflow.set_tracking_uri(uri=os.environ.get("MLFLOW_BACKEND"))
    timestamp = datetime.now().strftime('%Y-%m-%d-%H%M%S')
    experiment_name = f"exp-coronary-{timestamp}"
    
    mlflow.create_experiment(experiment_name)
    mlflow.set_experiment(experiment_name=experiment_name)
    
    with mlflow.start_run(run_name=f"run-{timestamp}"):
        for params in itertools.product(*parameters.values()):
            config = Config(*params)
            trainer = TrainingManager(config, transform=transforms.Resize(resolution))
            trainer.train()

def main():
    load_dotenv()
    
    args = parse_arguments()
    resolution = get_resolution(args)
    parameters = get_training_parameters(args)
    print("Training parameters:")
    print(parameters)
    print(f"Resolution: {resolution}")
    if not args.dryrun:
        run_training(parameters, resolution)

if __name__ == '__main__':
    main()