"""
TORCH-ARCADE Data Loader Integration for MLManager
Integrates the ARCADE dataset for coronary artery analysis with MLManager platform.

Based on: https://github.com/szuhow/torch-arcade
Dataset: ARCADE (Automatic Region-based Coronary Artery Disease Diagnostics)
"""

import os
import os.path
from pathlib import Path
from typing import Any, Callable, Optional, Tuple, Union, List
import logging

import numpy as np
import torch
from PIL import Image

try:
    from pycocotools.coco import COCO
    from pycocotools import mask as coco_mask
    COCO_AVAILABLE = True
except ImportError:
    COCO_AVAILABLE = False
    logging.warning("pycocotools not available. ARCADE dataset will not work properly.")

from torchvision.datasets.utils import download_and_extract_archive
from torchvision.datasets import VisionDataset
from torchvision import transforms
from torch.utils.data import DataLoader

# Import original torch-arcade encoding
from .encoding import ENCODING, COLOR_DICT, ARCADE_ENCODING

logger = logging.getLogger(__name__)

def distinguish_side(segments):
    """Determine if artery segments belong to right or left side"""
    return "right" if any(seg in segments for seg in ["1", "2", "3", "4", "16a", "16b", "16c"]) else "left"

def onehot_to_rgb(onehot, color_dict=COLOR_DICT):
    """Convert one-hot mask to RGB visualization"""
    single_layer = np.argmax(onehot, axis=-1)
    width, height, _ = onehot.shape
    output = np.zeros((width, height, 3), dtype=np.uint8)
    for k in range(len(color_dict)):
        output[single_layer == k] = np.array(color_dict[k])
    return output

def cached_mask(coco, cache_dir, 
                img_filename, img_id, 
                reduce, bg=True):
    """Original cached_mask function from torch-arcade"""
    img = Image.open(img_filename)
    mask_file = os.path.join(
        cache_dir,
        f"{os.path.basename(img_filename).replace('.png', '.npz')}"
    )
    categories = coco.loadCats(coco.getCatIds())

    if os.path.exists(mask_file):
        mask = np.load(mask_file)["mask"]
    else:
        annotations = coco.loadAnns(coco.getAnnIds(imgIds=img_id))
        mask = None
        for ann in annotations:
            category = categories[ann["category_id"]-1]
            mask = reduce(mask, coco.annToMask(ann), category)    
        if bg:
            bg_channel = (np.sum(mask, axis=-1) == 0).astype(np.uint8)
            bg_channel = bg_channel[..., np.newaxis]
            mask = np.concatenate([bg_channel, mask], axis=-1)
        np.savez_compressed(mask_file, mask=mask)
    return mask

def onehot_encode(class_idx, num_classes):
    """Create one-hot encoded vector for a given class index"""
    vector = np.zeros(num_classes, dtype=np.uint8)
    if 0 <= class_idx < num_classes:
        vector[class_idx] = 1
    return vector

class _ARCADEBase(VisionDataset):
    """
    Base ARCADE Dataset - original torch-arcade implementation
    """
    URL = "https://zenodo.org/records/8386059/files/arcade_challenge_datasets.zip"
    ZIPNAME = "arcade_challenge_datasets.zip"
    FILENAME = "arcade_challenge_datasets"
    
    DATASET_DICT = {
        "segmentation": {
            "train": {
                "path": os.path.join("dataset_phase_1", "segmentation_dataset", "seg_train"),
                "coco": "seg_train.json",
            },
            "val": {
                "path": os.path.join("dataset_phase_1", "segmentation_dataset", "seg_val"),
                "coco": "seg_val.json"
            },
            "test": {
                "path": os.path.join("dataset_final_phase", "test_case_segmentation"),
                "coco": "instances_default.json"
            }
        },
        "stenosis": {
            "train": {
                "path": os.path.join("dataset_phase_1", "stenosis_dataset", "sten_train"),
                "coco": "sten_train.json"
            },
            "val": {
                "path": os.path.join("dataset_phase_1", "stenosis_dataset", "sten_val"),
                "coco": "sten_val.json"
            },
            "test": {
                "path": os.path.join("dataset_final_phase", "test_cases_stenosis"),
                "coco": "instances_default.json"
            }
        },
    }

    def __init__(
        self,
        root: Union[str, Path],
        image_set: str = "train",
        task: str = "segmentation",
        side: str = None,
        download: bool = False,
        transform: Optional[Callable] = None,
        target_transform: Optional[Callable] = None,
        transforms: Optional[Callable] = None,
    ):
        super().__init__(root, transforms, transform, target_transform)

        if download:
            from torchvision.datasets.utils import download_and_extract_archive
            download_and_extract_archive(_ARCADEBase.URL, self.root, filename=_ARCADEBase.ZIPNAME)

        task_dict = _ARCADEBase.DATASET_DICT[task][image_set]
        self.dataset_dir = os.path.join(self.root, _ARCADEBase.FILENAME, task_dict["path"])
        self.coco = COCO(os.path.join(self.dataset_dir, "annotations", task_dict["coco"]))
        image_dir = os.path.join(self.dataset_dir, "images")

        self.images = []

        for image in self.coco.dataset['images']:
            img_id = image['id']
            annotations = self.coco.loadAnns(self.coco.getAnnIds(imgIds=img_id))
            segments = {self.coco.cats[ann["category_id"]]["name"] for ann in annotations}
            this_side = distinguish_side(segments)
            if side is None or this_side == side:
                file_path = os.path.join(image_dir, image['file_name'])
                if os.path.exists(file_path) and file_path.endswith('.png'):
                    self.images.append(file_path)

        self.file_to_id = {
            os.path.join(image_dir, image['file_name']): image['id'] 
            for image in self.coco.dataset['images']
        }

    def __len__(self) -> int:
        return len(self.images)
    
    # Dataset URLs and configuration
    URL = "https://zenodo.org/records/8386059/files/arcade_challenge_datasets.zip"
    ZIPNAME = "arcade_challenge_datasets.zip"
    FILENAME = "arcade_challenge_datasets"
    
    DATASET_CONFIG = {
        "segmentation": {
            "train": {
                "path": os.path.join("dataset_phase_1", "segmentation_dataset", "seg_train"),
                "coco": "seg_train.json",
            },
            "val": {
                "path": os.path.join("dataset_phase_1", "segmentation_dataset", "seg_val"),
                "coco": "seg_val.json"
            },
            "test": {
                "path": os.path.join("dataset_final_phase", "test_case_segmentation"),
                "coco": "instances_default.json"
            }
        },
        "stenosis": {
            "train": {
                "path": os.path.join("dataset_phase_1", "stenosis_dataset", "sten_train"),
                "coco": "sten_train.json"
            },
            "val": {
                "path": os.path.join("dataset_phase_1", "stenosis_dataset", "sten_val"),
                "coco": "sten_val.json"
            },
            "test": {
                "path": os.path.join("dataset_final_phase", "test_cases_stenosis"),
                "coco": "instances_default.json"
            }
        }
    }
    
    def __init__(
        self,
        root: Union[str, Path],
        image_set: str = "train",
        task: str = "segmentation",
        side: Optional[str] = None,
        download: bool = False,
        transform: Optional[Callable] = None,
        target_transform: Optional[Callable] = None,
        transforms: Optional[Callable] = None,
    ):
        """
        Initialize ARCADE dataset
        
        Args:
            root: Root directory for dataset
            image_set: 'train', 'val', or 'test'
            task: 'segmentation' or 'stenosis'
            side: Filter by artery side ('left', 'right', or None for both)
            download: Whether to download dataset if not found
            transform: Transform for input images
            target_transform: Transform for target masks
            transforms: Joint transforms for both image and target
        """
        super().__init__(root, transforms, transform, target_transform)
        
        if not COCO_AVAILABLE:
            raise ImportError("pycocotools is required for ARCADE dataset. Install with: pip install pycocotools")
        
        self.task = task
        self.side = side
        self.image_set = image_set
        
        # Setup dataset paths
        if download:
            self._download_dataset()
        
        # Load COCO annotations
        task_config = self.DATASET_CONFIG[task][image_set]
        self.dataset_dir = os.path.join(self.root, self.FILENAME, task_config["path"])
        coco_path = os.path.join(self.dataset_dir, "annotations", task_config["coco"])
        
        if not os.path.exists(coco_path):
            raise FileNotFoundError(f"COCO annotations not found at {coco_path}. Set download=True to download the dataset.")
        
        self.coco = COCO(coco_path)
        image_dir = os.path.join(self.dataset_dir, "images")
        
        # Filter images by side if specified
        self.images = []
        self.file_to_id = {}
        
        for image in self.coco.dataset['images']:
            img_id = image['id']
            file_path = os.path.join(image_dir, image['file_name'])
            
            if not os.path.exists(file_path) or not file_path.endswith('.png'):
                continue
            
            # Filter by artery side if requested
            if self.side is not None:
                annotations = self.coco.loadAnns(self.coco.getAnnIds(imgIds=img_id))
                segments = {self.coco.cats[ann["category_id"]]["name"] for ann in annotations}
                image_side = distinguish_side(segments)
                
                if image_side != self.side:
                    continue
            
            self.images.append(file_path)
            self.file_to_id[file_path] = img_id
        
        logger.info(f"Loaded ARCADE {task} dataset: {len(self.images)} images ({image_set} set)")
        if self.side:
            logger.info(f"Filtered for {self.side} artery side")
    
    def _download_dataset(self):
        """Download ARCADE dataset from Zenodo"""
        from torchvision.datasets.utils import download_and_extract_archive
        
        try:
            download_and_extract_archive(
                self.URL, 
                self.root, 
                filename=self.ZIPNAME,
                remove_finished=True
            )
            logger.info("ARCADE dataset downloaded successfully")
        except Exception as e:
            logger.error(f"Failed to download ARCADE dataset: {e}")
            raise
    
    def __len__(self) -> int:
        return len(self.images)
    
    def __getitem__(self, index: int) -> Tuple[Any, Any]:
        """Base implementation - override in subclasses"""
        raise NotImplementedError("Subclasses must implement __getitem__")

class ARCADEBinarySegmentation(_ARCADEBase):
    """
    ARCADE Binary Segmentation Dataset
    Returns: (image, binary_mask) where binary_mask indicates presence of any coronary artery
    """
    
    def __init__(self, *args, **kwargs):
        kwargs['task'] = 'segmentation'
        super().__init__(*args, **kwargs)
        
        # Create mask cache directory
        self.mask_cache_dir = os.path.join(self.dataset_dir, "masks_binary_cache")
        os.makedirs(self.mask_cache_dir, exist_ok=True)
    
    def _get_cached_mask(self, img_filename: str, img_id: int) -> np.ndarray:
        """Get or create cached binary mask"""
        cache_file = os.path.join(
            self.mask_cache_dir,
            f"{os.path.basename(img_filename).replace('.png', '.npz')}"
        )
        
        if os.path.exists(cache_file):
            return np.load(cache_file)["mask"]
        
        # Create binary mask from COCO annotations
        img = Image.open(img_filename)
        mask = np.zeros((img.height, img.width), dtype=np.uint8)
        
        annotations = self.coco.loadAnns(self.coco.getAnnIds(imgIds=img_id))
        for ann in annotations:
            # Convert COCO annotation to mask
            rle = coco_mask.frPyObjects(ann['segmentation'], img.height, img.width)
            binary_mask = coco_mask.decode(rle)
            if len(binary_mask.shape) == 3:
                binary_mask = binary_mask[:, :, 0]
            mask = mask | binary_mask
        
        # Scale mask from [0,1] to [0,255] for proper visualization
        mask = mask * 255
        
        # Cache the mask
        np.savez_compressed(cache_file, mask=mask)
        return mask
    
    def __getitem__(self, index: int) -> Tuple[torch.Tensor, torch.Tensor]:
        img_filename = self.images[index]
        img_id = self.file_to_id[img_filename]
        
        # Load image
        image = Image.open(img_filename).convert('RGB')
        
        # Get binary mask
        mask = self._get_cached_mask(img_filename, img_id)
        mask = Image.fromarray(mask)
        
        # Apply transforms
        if self.transforms is not None:
            image, mask = self.transforms(image, mask)
        elif self.transform is not None:
            image = self.transform(image)
            if self.target_transform is not None:
                mask = self.target_transform(mask)
        
        return image, mask

class ARCADEArteryClassification(_ARCADEBase):
    """
    ARCADE Artery Classification Dataset
    Input: image binary mask
    Label: 0 - right artery, 1 - left artery
    """
    
    def __init__(self, *args, **kwargs):
        kwargs['task'] = 'segmentation'  # Uses segmentation data
        super().__init__(*args, **kwargs)
        
        # Create binary mask cache directory
        self.mask_cache_dir = os.path.join(self.dataset_dir, "masks_binary_cache")
        os.makedirs(self.mask_cache_dir, exist_ok=True)
    
    def _get_cached_mask(self, img_filename: str, img_id: int) -> np.ndarray:
        """Get or create cached binary mask"""
        cache_file = os.path.join(
            self.mask_cache_dir,
            f"{os.path.basename(img_filename).replace('.png', '.npz')}"
        )
        
        if os.path.exists(cache_file):
            return np.load(cache_file)["mask"]
        
        # Create binary mask from COCO annotations
        img = Image.open(img_filename)
        mask = np.zeros((img.height, img.width), dtype=np.uint8)
        
        annotations = self.coco.loadAnns(self.coco.getAnnIds(imgIds=img_id))
        for ann in annotations:
            # Convert COCO annotation to mask
            rle = coco_mask.frPyObjects(ann['segmentation'], img.height, img.width)
            binary_mask = coco_mask.decode(rle)
            if len(binary_mask.shape) == 3:
                binary_mask = binary_mask[:, :, 0]
            mask = mask | binary_mask
        
        # Scale mask from [0,1] to [0,255] for proper visualization
        mask = mask * 255
        
        # Cache the mask
        np.savez_compressed(cache_file, mask=mask)
        return mask
    
    def _determine_artery_side(self, img_id: int) -> int:
        """Determine artery side from annotations"""
        annotations = self.coco.loadAnns(self.coco.getAnnIds(imgIds=img_id))
        categories = self.coco.loadCats(self.coco.getCatIds())
        
        segment_names = []
        for ann in annotations:
            category = categories[ann["category_id"] - 1]
            segment_names.append(category['name'])
        
        # Use the distinguish_side function
        side = distinguish_side(segment_names)
        return 0 if side == "right" else 1  # 0 for right, 1 for left
    
    def __getitem__(self, index: int) -> Tuple[torch.Tensor, torch.Tensor]:
        img_filename = self.images[index]
        img_id = self.file_to_id[img_filename]
        
        # Get binary mask as input
        mask = self._get_cached_mask(img_filename, img_id)
        mask_image = Image.fromarray(mask)
        
        # Get artery side classification label
        side_label = self._determine_artery_side(img_id)
        side_tensor = torch.tensor(side_label, dtype=torch.long)
        
        # Apply transforms
        if self.transforms is not None:
            mask_image, side_tensor = self.transforms(mask_image, side_tensor)
        elif self.transform is not None:
            mask_image = self.transform(mask_image)
        
        return mask_image, side_tensor


class ARCADESemanticSegmentation(_ARCADEBase):
    """
    ARCADE Semantic Segmentation Dataset
    Returns: (image, semantic_mask) where semantic_mask has 27 channels (one per artery segment + background)
    """
    
    def __init__(self, *args, **kwargs):
        kwargs['task'] = 'segmentation'
        super().__init__(*args, **kwargs)
        
        # Create mask cache directory
        self.mask_cache_dir = os.path.join(self.dataset_dir, "masks_semantic_cache")
        os.makedirs(self.mask_cache_dir, exist_ok=True)
    
    def _get_cached_mask(self, img_filename: str, img_id: int) -> np.ndarray:
        """Get or create cached semantic mask with one-hot encoding"""
        cache_file = os.path.join(
            self.mask_cache_dir,
            f"{os.path.basename(img_filename).replace('.png', '.npz')}"
        )
        
        if os.path.exists(cache_file):
            return np.load(cache_file)["mask"]
        
        # Create semantic mask from COCO annotations
        img = Image.open(img_filename)
        mask = np.zeros((img.height, img.width, 27), dtype=np.uint8)
        
        # Add background channel
        mask[:, :, 0] = 1
        
        annotations = self.coco.loadAnns(self.coco.getAnnIds(imgIds=img_id))
        categories = self.coco.loadCats(self.coco.getCatIds())
        
        for ann in annotations:
            category = categories[ann["category_id"] - 1]
            category_name = category['name']
            
            if category_name in ARCADE_ENCODING:
                # Convert COCO annotation to mask
                rle = coco_mask.frPyObjects(ann['segmentation'], img.height, img.width)
                binary_mask = coco_mask.decode(rle)
                if len(binary_mask.shape) == 3:
                    binary_mask = binary_mask[:, :, 0]
                
                # Apply one-hot encoding
                class_idx = ARCADE_ENCODING[category_name]
                one_hot_vector = onehot_encode(class_idx, 27)
                
                # Update mask where annotation is present (binary_mask values are [0,1])
                mask[binary_mask == 1] = one_hot_vector
        
        # Cache the mask
        np.savez_compressed(cache_file, mask=mask)
        return mask
    
    def __getitem__(self, index: int) -> Tuple[torch.Tensor, torch.Tensor]:
        img_filename = self.images[index]
        img_id = self.file_to_id[img_filename]
        
        # Load image
        image = Image.open(img_filename).convert('RGB')
        
        # Get semantic mask
        mask = self._get_cached_mask(img_filename, img_id)
        
        # Apply transforms
        if self.transforms is not None:
            image, mask = self.transforms(image, mask)
        elif self.transform is not None:
            image = self.transform(image)
            if self.target_transform is not None:
                mask = self.target_transform(mask)
        
        return image, mask

class ARCADESemanticSegmentationBinary(_ARCADEBase):
    """
    ARCADE Semantic Segmentation Binary Dataset
    Input: image binary mask
    Label: image semantic mask (shape 512x512x26)
    """
    
    def __init__(self, *args, **kwargs):
        kwargs['task'] = 'segmentation'
        super().__init__(*args, **kwargs)
        
        # Create cache directories
        self.mask_cache_dir = os.path.join(self.dataset_dir, "masks_binary_cache")
        self.semantic_cache_dir = os.path.join(self.dataset_dir, "masks_semantic_cache")
        os.makedirs(self.mask_cache_dir, exist_ok=True)
        os.makedirs(self.semantic_cache_dir, exist_ok=True)
    
    def _get_cached_binary_mask(self, img_filename: str, img_id: int) -> np.ndarray:
        """Get or create cached binary mask"""
        cache_file = os.path.join(
            self.mask_cache_dir,
            f"{os.path.basename(img_filename).replace('.png', '.npz')}"
        )
        
        if os.path.exists(cache_file):
            return np.load(cache_file)["mask"]
        
        # Create binary mask from COCO annotations
        img = Image.open(img_filename)
        mask = np.zeros((img.height, img.width), dtype=np.uint8)
        
        annotations = self.coco.loadAnns(self.coco.getAnnIds(imgIds=img_id))
        for ann in annotations:
            # Convert COCO annotation to mask
            rle = coco_mask.frPyObjects(ann['segmentation'], img.height, img.width)
            binary_mask = coco_mask.decode(rle)
            if len(binary_mask.shape) == 3:
                binary_mask = binary_mask[:, :, 0]
            mask = mask | binary_mask
        
        # Scale mask from [0,1] to [0,255] for proper visualization
        mask = mask * 255
        
        # Cache the mask
        np.savez_compressed(cache_file, mask=mask)
        return mask
    
    def _get_cached_semantic_mask(self, img_filename: str, img_id: int) -> np.ndarray:
        """Get or create cached semantic mask with 26 channels (excluding background)"""
        cache_file = os.path.join(
            self.semantic_cache_dir,
            f"{os.path.basename(img_filename).replace('.png', '.npz')}"
        )
        
        if os.path.exists(cache_file):
            return np.load(cache_file)["mask"]
        
        # Create semantic mask from COCO annotations
        img = Image.open(img_filename)
        mask = np.zeros((img.height, img.width, 26), dtype=np.uint8)  # 26 channels (no background)
        
        annotations = self.coco.loadAnns(self.coco.getAnnIds(imgIds=img_id))
        categories = self.coco.loadCats(self.coco.getCatIds())
        
        for ann in annotations:
            category = categories[ann["category_id"] - 1]
            category_name = category['name']
            
            if category_name in ARCADE_ENCODING and category_name != "background":
                # Convert COCO annotation to mask
                rle = coco_mask.frPyObjects(ann['segmentation'], img.height, img.width)
                binary_mask = coco_mask.decode(rle)
                if len(binary_mask.shape) == 3:
                    binary_mask = binary_mask[:, :, 0]
                
                # Apply encoding (subtract 1 to exclude background channel)
                class_idx = ARCADE_ENCODING[category_name] - 1  # 0-25 instead of 1-26
                if 0 <= class_idx < 26:
                    mask[:, :, class_idx][binary_mask == 1] = 1
        
        # Cache the mask
        np.savez_compressed(cache_file, mask=mask)
        return mask
    
    def __getitem__(self, index: int) -> Tuple[torch.Tensor, torch.Tensor]:
        img_filename = self.images[index]
        img_id = self.file_to_id[img_filename]
        
        # Get binary mask as input
        binary_mask = self._get_cached_binary_mask(img_filename, img_id)
        binary_mask_image = Image.fromarray(binary_mask)
        
        # Get semantic mask as label
        semantic_mask = self._get_cached_semantic_mask(img_filename, img_id)
        
        # Apply transforms
        if self.transforms is not None:
            binary_mask_image, semantic_mask = self.transforms(binary_mask_image, semantic_mask)
        elif self.transform is not None:
            binary_mask_image = self.transform(binary_mask_image)
            if self.target_transform is not None:
                semantic_mask = self.target_transform(semantic_mask)
        
        return binary_mask_image, semantic_mask


class ARCADEStenosisDetection(_ARCADEBase):
    """
    ARCADE Stenosis Detection Dataset
    Returns: (image, annotations) for object detection of stenoses
    """
    
    def __init__(self, *args, **kwargs):
        kwargs['task'] = 'stenosis'
        super().__init__(*args, **kwargs)
    
    def __getitem__(self, index: int) -> Tuple[torch.Tensor, List]:
        img_filename = self.images[index]
        img_id = self.file_to_id[img_filename]
        
        # Load image
        image = Image.open(img_filename).convert('RGB')
        
        # Get stenosis annotations
        annotations = self.coco.loadAnns(self.coco.getAnnIds(imgIds=img_id))
        
        # Apply transforms
        if self.transforms is not None:
            image, annotations = self.transforms(image, annotations)
        elif self.transform is not None:
            image = self.transform(image)
        
        return image, annotations

class ARCADEStenosisSegmentation(_ARCADEBase):
    """
    ARCADE Stenosis Segmentation Dataset
    Input: image
    Label: binary mask (0, 1) for stenoses
    """
    
    def __init__(self, *args, **kwargs):
        kwargs['task'] = 'stenosis'
        super().__init__(*args, **kwargs)
        
        # Create mask cache directory
        self.mask_cache_dir = os.path.join(self.dataset_dir, "masks_stenosis_cache")
        os.makedirs(self.mask_cache_dir, exist_ok=True)
    
    def _get_cached_stenosis_mask(self, img_filename: str, img_id: int) -> np.ndarray:
        """Get or create cached stenosis binary mask"""
        cache_file = os.path.join(
            self.mask_cache_dir,
            f"{os.path.basename(img_filename).replace('.png', '.npz')}"
        )
        
        if os.path.exists(cache_file):
            return np.load(cache_file)["mask"]
        
        # Create stenosis mask from COCO annotations
        img = Image.open(img_filename)
        mask = np.zeros((img.height, img.width), dtype=np.uint8)
        
        annotations = self.coco.loadAnns(self.coco.getAnnIds(imgIds=img_id))
        categories = self.coco.loadCats(self.coco.getCatIds())
        
        for ann in annotations:
            category = categories[ann["category_id"] - 1]
            category_name = category['name']
            
            # Only process stenosis annotations
            if category_name == "stenosis":
                # Convert COCO annotation to mask
                rle = coco_mask.frPyObjects(ann['segmentation'], img.height, img.width)
                binary_mask = coco_mask.decode(rle)
                if len(binary_mask.shape) == 3:
                    binary_mask = binary_mask[:, :, 0]
                mask = mask | binary_mask
        
        # Scale mask from [0,1] to [0,255] for proper visualization
        mask = mask * 255
        
        # Cache the mask
        np.savez_compressed(cache_file, mask=mask)
        return mask
    
    def __getitem__(self, index: int) -> Tuple[torch.Tensor, torch.Tensor]:
        img_filename = self.images[index]
        img_id = self.file_to_id[img_filename]
        
        # Load image
        image = Image.open(img_filename).convert('RGB')
        
        # Get stenosis mask
        mask = self._get_cached_stenosis_mask(img_filename, img_id)
        mask_image = Image.fromarray(mask)
        
        # Apply transforms
        if self.transforms is not None:
            image, mask_image = self.transforms(image, mask_image)
        elif self.transform is not None:
            image = self.transform(image)
            if self.target_transform is not None:
                mask_image = self.target_transform(mask_image)
        
        return image, mask_image

def create_arcade_dataloader(
    root: str,
    task: str = "binary_segmentation",
    image_set: str = "train",
    batch_size: int = 4,
    num_workers: int = 2,
    shuffle: bool = True,
    download: bool = False,
    image_size: int = 512,
    side: Optional[str] = None,
    **kwargs
) -> DataLoader:
    """
    Create ARCADE DataLoader for MLManager integration - supports all 6 dataset types
    
    Args:
        root: Dataset root directory
        task: Type of task - one of:
            - 'binary_segmentation': image → binary mask
            - 'semantic_segmentation': image → 27-class semantic mask  
            - 'artery_classification': binary mask → left/right classification
            - 'semantic_segmentation_binary': binary mask → 26-class semantic mask
            - 'stenosis_detection': image → COCO bounding boxes
            - 'stenosis_segmentation': image → stenosis binary mask
        image_set: Dataset split ('train', 'val', 'test')
        batch_size: Batch size for training
        num_workers: Number of worker processes
        shuffle: Whether to shuffle data
        download: Download dataset if not found
        image_size: Target image size (will be resized)
        side: Filter by artery side ('left', 'right', or None)
        **kwargs: Additional arguments for DataLoader
    
    Returns:
        DataLoader: Configured PyTorch DataLoader for the specified ARCADE task
    """
    
    # Define transforms
    if task in ["binary_segmentation", "semantic_segmentation"]:
        # Transforms for segmentation tasks
        image_transform = transforms.Compose([
            transforms.Resize((image_size, image_size)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
        
        if task == "binary_segmentation":
            mask_transform = transforms.Compose([
                transforms.Resize((image_size, image_size)),
                transforms.ToTensor()
            ])
        else:  # semantic_segmentation
            # For semantic segmentation, we handle the transform differently
            mask_transform = transforms.Compose([
                transforms.Lambda(lambda x: torch.from_numpy(x).permute(2, 0, 1).float())
            ])
    else:  # stenosis_detection
        image_transform = transforms.Compose([
            transforms.Resize((image_size, image_size)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
        mask_transform = None
    
    # Create dataset - support all 6 ARCADE dataset types
    if task == "binary_segmentation":
        dataset = ARCADEBinarySegmentation(
            root=root,
            image_set=image_set,
            side=side,
            download=download,
            transform=image_transform,
            target_transform=mask_transform
        )
    elif task == "semantic_segmentation":
        dataset = ARCADESemanticSegmentation(
            root=root,
            image_set=image_set,
            side=side,
            download=download,
            transform=image_transform,
            target_transform=mask_transform
        )
    elif task == "artery_classification":
        dataset = ARCADEArteryClassification(
            root=root,
            image_set=image_set,
            download=download,
            transform=image_transform,
            target_transform=mask_transform
        )
    elif task == "semantic_segmentation_binary":
        dataset = ARCADESemanticSegmentationBinary(
            root=root,
            image_set=image_set,
            side=side,
            download=download,
            transform=image_transform,
            target_transform=mask_transform
        )
    elif task == "stenosis_detection":
        dataset = ARCADEStenosisDetection(
            root=root,
            image_set=image_set,
            side=side,
            download=download,
            transform=image_transform
        )
    elif task == "stenosis_segmentation":
        dataset = ARCADEStenosisSegmentation(
            root=root,
            image_set=image_set,
            side=side,
            download=download,
            transform=image_transform,
            target_transform=mask_transform
        )
    else:
        raise ValueError(f"Unknown task: {task}. Supported tasks: binary_segmentation, semantic_segmentation, artery_classification, semantic_segmentation_binary, stenosis_detection, stenosis_segmentation")
    
    # Create DataLoader
    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        **kwargs
    )
    
    logger.info(f"Created ARCADE DataLoader: {task}, {image_set} set, {len(dataset)} samples")
    
    return dataloader

# Integration functions for MLManager
def get_arcade_dataset_info(root: str, download: bool = False) -> dict:
    """Get information about available ARCADE datasets"""
    info = {
        "dataset_name": "ARCADE",
        "description": "Automatic Region-based Coronary Artery Disease Diagnostics",
        "tasks": ["binary_segmentation", "semantic_segmentation", "stenosis_detection"],
        "splits": ["train", "val", "test"],
        "sides": ["left", "right", "both"],
        "num_classes": {
            "binary_segmentation": 2,
            "semantic_segmentation": 27,
            "stenosis_detection": 1
        },
        "image_size": (512, 512),
        "available": False
    }
    
    # Check if dataset is available
    dataset_path = os.path.join(root, ARCADEDataset.FILENAME)
    if os.path.exists(dataset_path):
        info["available"] = True
        
        # Count samples if available
        try:
            for task in ["segmentation", "stenosis"]:
                for split in ["train", "val"]:
                    try:
                        if task == "segmentation":
                            dataset = ARCADEBinarySegmentation(root, split, download=False)
                        else:
                            dataset = ARCADEStenosisDetection(root, split, download=False)
                        info[f"{task}_{split}_samples"] = len(dataset)
                    except:
                        pass
        except:
            pass
    
    return info

if __name__ == "__main__":
    # Example usage
    print("ARCADE Dataset Integration for MLManager")
    print("=" * 50)
    
    # Test dataset info
    root = "./datasets"
    info = get_arcade_dataset_info(root)
    print("Dataset Info:", info)
    
    # Test data loader creation
    try:
        dataloader = create_arcade_dataloader(
            root=root,
            task="binary_segmentation",
            image_set="train",
            batch_size=2,
            download=True,  # Set to True to download
            image_size=256
        )
        
        print(f"DataLoader created successfully: {len(dataloader)} batches")
        
        # Test one batch
        for batch_idx, (images, masks) in enumerate(dataloader):
            print(f"Batch {batch_idx}: Images {images.shape}, Masks {masks.shape}")
            break
            
    except Exception as e:
        print(f"Error creating dataloader: {e}")
