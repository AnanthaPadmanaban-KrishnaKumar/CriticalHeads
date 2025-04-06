"""
Data loading utilities for the SAM benchmark.

This module provides dataset-specific data loader classes for various semantic segmentation
datasets, handling their native formats and official splits appropriately.
"""

import os
from abc import ABC, abstractmethod
from typing import Dict, List, Optional, Tuple, Union

import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from PIL import Image

# Import transforms that will be applied consistently across datasets
# from data.transforms import SAMTransform
from tqdm import tqdm  # Added progress bar support


class BaseDataset(Dataset, ABC):
    """Abstract base class for all dataset implementations."""
    
    def __init__(
        self,
        root_dir: str,
        split: str = "val",
        transform=None,
        target_transform=None,
    ):
        """
        Initialize the base dataset.
        
        Args:
            root_dir: Path to the dataset root directory
            split: Dataset split ('train', 'val', 'test')
            transform: Image transformations to apply
            target_transform: Mask/annotation transformations to apply
        """
        self.root_dir = root_dir
        self.split = split
        self.transform = transform
        self.target_transform = target_transform
        
        # Validate split
        if split not in self.available_splits():
            raise ValueError(f"Split '{split}' not available for {self.__class__.__name__}. "
                             f"Available splits: {self.available_splits()}")
        
        # Set up data paths and load file list
        self.setup()
        
    @abstractmethod
    def setup(self) -> None:
        """Set up dataset-specific paths and file lists."""
        pass
    
    @classmethod
    @abstractmethod
    def available_splits(cls) -> List[str]:
        """Return a list of available dataset splits."""
        pass
    
    @abstractmethod
    def __len__(self) -> int:
        """Return the total number of samples in the dataset."""
        pass
    
    @abstractmethod
    def __getitem__(self, idx: int) -> Dict:
        """
        Get an item from the dataset.
        
        Returns:
            A dictionary containing at minimum:
            - 'image': The preprocessed image tensor
            - 'mask': Ground truth segmentation mask
            - 'image_id': Unique identifier for the image
            - 'original_size': Original size of the image before transformations
        """
        pass
    
    def get_categories(self) -> List[Dict]:
        """
        Get dataset category information.
        
        Returns:
            A list of dictionaries containing category information
        """
        raise NotImplementedError("This dataset does not provide category information")


class CityscapesDataset(BaseDataset):
    """Loader for the Cityscapes dataset."""
    
    # Standard Cityscapes classes
    CLASSES = [
        'road', 'sidewalk', 'building', 'wall', 'fence', 'pole', 'traffic light',
        'traffic sign', 'vegetation', 'terrain', 'sky', 'person', 'rider', 'car',
        'truck', 'bus', 'train', 'motorcycle', 'bicycle'
    ]
    
    @classmethod
    def available_splits(cls) -> List[str]:
        return ["train", "val", "test"]
    
    def setup(self) -> None:
        """Set up Cityscapes-specific paths and file lists."""
        self.images_dir = os.path.join(self.root_dir, "leftImg8bit", self.split)
        self.masks_dir = os.path.join(self.root_dir, "gtFine", self.split)
        
        # Get list of all images
        self.images = []
        self.masks = []
        
        for city in tqdm(os.listdir(self.images_dir), desc="Scanning Cityscapes Cities"):
            city_img_dir = os.path.join(self.images_dir, city)
            city_mask_dir = os.path.join(self.masks_dir, city)
            # Skip if not a directory (e.g., .DS_Store)
            if not os.path.isdir(city_img_dir):
                continue
            for filename in os.listdir(city_img_dir):
                if filename.endswith("_leftImg8bit.png"):
                    image_path = os.path.join(city_img_dir, filename)
                    
                    # Construct corresponding mask filename
                    mask_filename = filename.replace("_leftImg8bit.png", "_gtFine_labelIds.png")
                    mask_path = os.path.join(city_mask_dir, mask_filename)
                    
                    if os.path.exists(mask_path):
                        self.images.append(image_path)
                        self.masks.append(mask_path)
    
    def __len__(self) -> int:
        return len(self.images)
    
    def __getitem__(self, idx: int) -> Dict:
        """Get a sample from the Cityscapes dataset."""
        image_path = self.images[idx]
        mask_path = self.masks[idx]
        
        # Load image and mask
        image = Image.open(image_path).convert("RGB")
        mask = Image.open(mask_path)
        
        # Store original size before any transformations
        original_size = image.size
        
        # Apply transformations if provided
        if self.transform:
            image = self.transform(image)
        
        if self.target_transform:
            mask = self.target_transform(mask)
        else:
            # Convert to numpy array
            mask = np.array(mask)
        
        # Extract image ID from filename
        image_id = os.path.basename(image_path).split("_leftImg8bit")[0]
        
        return {
            'image': image,
            'mask': mask,
            'image_id': image_id,
            'original_size': original_size,
            'image_path': image_path
        }
    
    def get_categories(self) -> List[Dict]:
        """Get Cityscapes category information."""
        categories = []
        for i, name in enumerate(self.CLASSES):
            categories.append({
                'id': i,
                'name': name,
                'supercategory': 'none'
            })
        return categories


class COCODataset(BaseDataset):
    """Loader for the COCO dataset."""
    
    @classmethod
    def available_splits(cls) -> List[str]:
        return ["train", "val"]
    
    def setup(self) -> None:
        """Set up COCO-specific paths and file lists."""
        from pycocotools.coco import COCO
        
        # Map split name to COCO file name
        split_map = {
            "train": "train2017",
            "val": "val2017"
        }
        
        self.images_dir = os.path.join(self.root_dir, split_map[self.split])
        anno_file = os.path.join(
            self.root_dir, 
            "annotations", 
            f"instances_{split_map[self.split]}.json"
        )
        
        self.coco = COCO(anno_file)
        self.img_ids = list(sorted(self.coco.imgs.keys()))
        
        # Filter out images without annotations, with progress bar
        if self.split != "test":
            filtered_ids = []
            for img_id in tqdm(self.img_ids, desc="Filtering COCO Image IDs"):
                ann_ids = self.coco.getAnnIds(imgIds=img_id)
                if len(ann_ids) > 0:
                    filtered_ids.append(img_id)
            self.img_ids = filtered_ids
    
    def __len__(self) -> int:
        return len(self.img_ids)
    
    def __getitem__(self, idx: int) -> Dict:
        """Get a sample from the COCO dataset."""
        img_id = self.img_ids[idx]
        image_info = self.coco.loadImgs(img_id)[0]
        image_path = os.path.join(self.images_dir, image_info['file_name'])
        
        # Load image
        image = Image.open(image_path).convert("RGB")
        original_size = image.size
        
        # Get annotations
        ann_ids = self.coco.getAnnIds(imgIds=img_id)
        anns = self.coco.loadAnns(ann_ids)
        
        # Create segmentation mask
        mask = np.zeros((image_info['height'], image_info['width']), dtype=np.int32)
        for i, ann in enumerate(anns):
            # Use category_id as the mask value
            category_id = ann['category_id']
            if 'segmentation' in ann:
                rle = self.coco.annToRLE(ann)
                m = self.coco.decode(rle)
                mask[m > 0] = category_id
        
        # Apply transformations
        if self.transform:
            image = self.transform(image)
        
        if self.target_transform:
            mask = self.target_transform(mask)
        
        return {
            'image': image,
            'mask': mask,
            'image_id': img_id,
            'original_size': original_size,
            'image_path': image_path
        }
    
    def get_categories(self) -> List[Dict]:
        """Get COCO category information."""
        return self.coco.loadCats(self.coco.getCatIds())


class ADE20KDataset(BaseDataset):
    """Loader for the ADE20K dataset."""
    
    @classmethod
    def available_splits(cls) -> List[str]:
        return ["train", "val"]
    
    def setup(self) -> None:
        """Set up ADE20K-specific paths and file lists."""
        self.images_dir = os.path.join(self.root_dir, "images", self.split)
        self.masks_dir = os.path.join(self.root_dir, "annotations", self.split)
        
        # Get list of all images using a progress bar
        self.images = []
        for root, _, files in tqdm(os.walk(self.images_dir), desc="Scanning ADE20K Images"):
            for file in files:
                if file.endswith(".jpg"):
                    self.images.append(os.path.join(root, file))
        
        # Sort for reproducibility
        self.images.sort()
        
        # Create corresponding mask paths
        self.masks = []
        for img_path in self.images:
            rel_path = os.path.relpath(img_path, self.images_dir)
            mask_path = os.path.join(
                self.masks_dir, 
                os.path.splitext(rel_path)[0] + ".png"
            )
            self.masks.append(mask_path)
    
    def __len__(self) -> int:
        return len(self.images)
    
    def __getitem__(self, idx: int) -> Dict:
        """Get a sample from the ADE20K dataset."""
        image_path = self.images[idx]
        mask_path = self.masks[idx]
        
        # Load image and mask
        image = Image.open(image_path).convert("RGB")
        mask = Image.open(mask_path)
        
        # Store original size
        original_size = image.size
        
        # Apply transformations
        if self.transform:
            image = self.transform(image)
        
        if self.target_transform:
            mask = self.target_transform(mask)
        else:
            mask = np.array(mask)
        
        # Extract image ID from filename
        image_id = os.path.basename(image_path).split(".")[0]
        
        return {
            'image': image,
            'mask': mask,
            'image_id': image_id,
            'original_size': original_size,
            'image_path': image_path
        }


class PascalVOCDataset(BaseDataset):
    """Loader for the Pascal VOC dataset."""
    
    # Pascal VOC classes
    CLASSES = [
        'background', 'aeroplane', 'bicycle', 'bird', 'boat', 'bottle', 'bus',
        'car', 'cat', 'chair', 'cow', 'diningtable', 'dog', 'horse', 'motorbike',
        'person', 'pottedplant', 'sheep', 'sofa', 'train', 'tvmonitor'
    ]
    
    @classmethod
    def available_splits(cls) -> List[str]:
        return ["train", "val", "trainval"]
    
    def setup(self) -> None:
        """Set up Pascal VOC specific paths and file lists."""
        self.images_dir = os.path.join(self.root_dir, "JPEGImages")
        self.masks_dir = os.path.join(self.root_dir, "SegmentationClass")
        
        # Read the split file
        split_file = os.path.join(
            self.root_dir, 
            "ImageSets", 
            "Segmentation", 
            f"{self.split}.txt"
        )
        
        with open(split_file, "r") as f:
            file_ids = [line.strip() for line in f]
        
        # Use progress bar for loading file IDs
        self.images = [os.path.join(self.images_dir, f"{file_id}.jpg") for file_id in tqdm(file_ids, desc="Loading Pascal VOC Image IDs")]
        self.masks = [os.path.join(self.masks_dir, f"{file_id}.png") for file_id in tqdm(file_ids, desc="Loading Pascal VOC Masks")]
    
    def __len__(self) -> int:
        return len(self.images)
    
    def __getitem__(self, idx: int) -> Dict:
        """Get a sample from the Pascal VOC dataset."""
        image_path = self.images[idx]
        mask_path = self.masks[idx]
        
        # Load image and mask
        image = Image.open(image_path).convert("RGB")
        mask = Image.open(mask_path)
        
        # Store original size
        original_size = image.size
        
        # Apply transformations
        if self.transform:
            image = self.transform(image)
        
        if self.target_transform:
            mask = self.target_transform(mask)
        else:
            mask = np.array(mask)
        
        # Extract image ID from filename
        image_id = os.path.basename(image_path).split(".")[0]
        
        return {
            'image': image,
            'mask': mask,
            'image_id': image_id,
            'original_size': original_size,
            'image_path': image_path
        }
    
    def get_categories(self) -> List[Dict]:
        """Get Pascal VOC category information."""
        categories = []
        for i, name in enumerate(self.CLASSES):
            categories.append({
                'id': i,
                'name': name,
                'supercategory': 'none'
            })
        return categories


class MapillaryVistasDataset(BaseDataset):
    """Loader for the Mapillary Vistas dataset."""
    
    @classmethod
    def available_splits(cls) -> List[str]:
        return ["training", "validation", "testing"]
    
    def setup(self) -> None:
        """Set up Mapillary Vistas specific paths and file lists."""
        # Map our standard split names to Mapillary's naming
        split_map = {
            "train": "training",
            "val": "validation",
            "test": "testing"
        }
        
        # Use the mapped split name or the original if not in map
        actual_split = split_map.get(self.split, self.split)
        
        self.images_dir = os.path.join(self.root_dir, actual_split, "images")
        self.masks_dir = os.path.join(self.root_dir, actual_split, "labels")
        
        # Get list of all images with progress bar
        self.images = [os.path.join(self.images_dir, f) 
                       for f in tqdm(os.listdir(self.images_dir), desc="Scanning Mapillary Images")
                       if f.endswith(('.jpg', '.jpeg', '.png'))]
        
        # Sort for reproducibility
        self.images.sort()
        
        # Create corresponding mask paths
        self.masks = []
        for img_path in self.images:
            img_filename = os.path.basename(img_path)
            base_filename, _ = os.path.splitext(img_filename)
            mask_path = os.path.join(self.masks_dir, f"{base_filename}.png")
            self.masks.append(mask_path)
    
    def __len__(self) -> int:
        return len(self.images)
    
    def __getitem__(self, idx: int) -> Dict:
        """Get a sample from the Mapillary Vistas dataset."""
        image_path = self.images[idx]
        mask_path = self.masks[idx]
        
        # Load image and mask
        image = Image.open(image_path).convert("RGB")
        mask = Image.open(mask_path)
        
        # Store original size
        original_size = image.size
        
        # Apply transformations
        if self.transform:
            image = self.transform(image)
        
        if self.target_transform:
            mask = self.target_transform(mask)
        else:
            mask = np.array(mask)
        
        # Extract image ID from filename
        image_id = os.path.basename(image_path).split(".")[0]
        
        return {
            'image': image,
            'mask': mask,
            'image_id': image_id,
            'original_size': original_size,
            'image_path': image_path
        }


# Factory function to get the appropriate dataset
def get_dataset(dataset_name: str, **kwargs) -> BaseDataset:
    """
    Factory function to create dataset instance based on name.
    
    Args:
        dataset_name: Name of the dataset ('cityscapes', 'coco', 'ade20k', etc.)
        **kwargs: Arguments to pass to the dataset constructor
    
    Returns:
        Instantiated dataset object
    """
    datasets = {
        'cityscapes': CityscapesDataset,
        'coco': COCODataset,
        'ade20k': ADE20KDataset,
        'pascal_voc': PascalVOCDataset,
        'mapillary_vistas': MapillaryVistasDataset,
        # Add more datasets as implemented
    }
    
    if dataset_name.lower() not in datasets:
        raise ValueError(f"Dataset '{dataset_name}' not supported. "
                         f"Available datasets: {list(datasets.keys())}")
    
    return datasets[dataset_name.lower()](**kwargs)


def get_dataloader(
    dataset_name: str,
    root_dir: str,
    split: str = "val",
    batch_size: int = 1,
    num_workers: int = 4,
    transform=None,
    target_transform=None,
    shuffle: bool = False,
    **kwargs
) -> DataLoader:
    """
    Create a DataLoader for the specified dataset.
    
    Args:
        dataset_name: Name of the dataset ('cityscapes', 'coco', 'ade20k', etc.)
        root_dir: Path to the dataset root directory
        split: Dataset split ('train', 'val', 'test')
        batch_size: Batch size for the DataLoader
        num_workers: Number of worker processes for data loading
        transform: Image transformations to apply
        target_transform: Mask/annotation transformations to apply
        shuffle: Whether to shuffle the dataset
        **kwargs: Additional arguments to pass to the dataset constructor
    
    Returns:
        DataLoader object for the specified dataset
    """
    dataset = get_dataset(
        dataset_name=dataset_name,
        root_dir=root_dir,
        split=split,
        transform=transform,
        target_transform=target_transform,
        **kwargs
    )
    
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=True
    )

if __name__ == "__main__":
    import argparse
    from torchvision.transforms import ToTensor  # Import ToTensor
    import matplotlib.pyplot as plt  # For displaying images

    # Parse command-line arguments for testing (optional)
    parser = argparse.ArgumentParser(description="Test DataLoader for a given dataset")
    parser.add_argument("--dataset", type=str, default="cityscapes", help="Name of the dataset to test")
    parser.add_argument("--root_dir", type=str, required=True, help="Root directory of the dataset")
    parser.add_argument("--split", type=str, default="val", help="Dataset split to use (e.g., train, val, test)")
    parser.add_argument("--batch_size", type=int, default=5, help="Batch size for the DataLoader")
    args = parser.parse_args()

    # Create the DataLoader using the factory function
    dataloader = get_dataloader(
        dataset_name=args.dataset,
        root_dir=args.root_dir,
        split=args.split,
        batch_size=args.batch_size,
        num_workers=0,  # For debugging, use 0 to avoid multiprocessing issues
        transform=ToTensor(),  # Convert PIL images to tensors
        target_transform=ToTensor(),  # Optional: convert mask to tensor (if appropriate)
        shuffle=False
    )

    # Get one batch from the DataLoader
    batch = next(iter(dataloader))
    images = batch['image']       # Tensor shape: (batch_size, 3, H, W)
    masks = batch['mask']         # Tensor shape: (batch_size, H, W) or (batch_size, 1, H, W)
    image_ids = batch['image_id'] # List of image IDs

    # Create a figure with 5 rows and 2 columns (Image | Mask)
    fig, axes = plt.subplots(nrows=args.batch_size, ncols=2, figsize=(10, 5 * args.batch_size))
    
    for i in range(args.batch_size):
        # Convert image tensor to numpy and change channel order from CxHxW to HxWxC
        img = images[i].permute(1, 2, 0).numpy()
        # Convert mask to numpy; if mask has a singleton channel, squeeze it
        mask = masks[i].numpy()
        if mask.ndim == 3 and mask.shape[0] == 1:
            mask = mask.squeeze(0)
        
        # Display image
        axes[i, 0].imshow(img)
        axes[i, 0].set_title(f"Image: {image_ids[i]}")
        axes[i, 0].axis("off")
        
        # Display mask
        axes[i, 1].imshow(mask, cmap="gray")
        axes[i, 1].set_title("Mask")
        axes[i, 1].axis("off")
    
    plt.tight_layout()
    plt.show()
