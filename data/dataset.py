import torch
from torch.utils.data import Dataset
import cv2
import numpy as np
import os
import albumentations as A
from albumentations.pytorch import ToTensorV2

class AnimateXDataset(Dataset):
    """
    Dataset class for the AnimateX model.
    
    This dataset loads and preprocesses the data required for training and evaluating
    the AnimateX model, including reference images, driving videos, pose sequences,
    and text prompts.
    """

    def __init__(self, data_dir, transform=None):
        """
        Initialize the AnimateXDataset.

        Args:
            data_dir (str): Path to the directory containing the dataset.
            transform (callable, optional): A function/transform to apply to the image data.
                If None, a default transform is applied.
        """
        self.data_dir = data_dir
        self.transform = transform or A.Compose([
            A.Resize(256, 256),
            A.HorizontalFlip(p=0.5),
            A.RandomBrightnessContrast(p=0.2),
            A.Normalize(),
            ToTensorV2()
        ])
        self.samples = self._load_samples()
    
    def _load_samples(self):
        """
        Load all valid samples from the data directory.

        Returns:
            list: A list of tuples, each containing paths to the reference image,
                  driving video, pose sequence, and the text prompt.
        """
        samples = []
        for sample_dir in os.listdir(self.data_dir):
            sample_path = os.path.join(self.data_dir, sample_dir)
            if os.path.isdir(sample_path):
                reference_image = os.path.join(sample_path, 'reference.png')
                driving_video = os.path.join(sample_path, 'driving.mp4')
                pose_sequence = os.path.join(sample_path, 'pose.npy')
                text_prompt = os.path.join(sample_path, 'prompt.txt')
                if all(os.path.exists(f) for f in [reference_image, driving_video, pose_sequence, text_prompt]):
                    with open(text_prompt, 'r') as f:
                        prompt = f.read().strip()
                    samples.append((reference_image, driving_video, pose_sequence, prompt))
        return samples
    
    def __len__(self):
        """
        Get the total number of samples in the dataset.

        Returns:
            int: The number of samples in the dataset.
        """
        return len(self.samples)
    
    def __getitem__(self, idx):
        """
        Get a single sample from the dataset.

        Args:
            idx (int): The index of the sample to retrieve.

        Returns:
            tuple: A tuple containing:
                - reference_image (torch.Tensor): The reference image.
                - driving_video (torch.Tensor): The driving video frames.
                - pose_sequence (torch.Tensor): The pose sequence.
                - text_prompt (str): The text prompt.
        """
        reference_image_path, driving_video_path, pose_sequence_path, text_prompt = self.samples[idx]
        
        # Load reference image
        reference_image = cv2.imread(reference_image_path)
        reference_image = cv2.cvtColor(reference_image, cv2.COLOR_BGR2RGB)
        
        # Load driving video
        cap = cv2.VideoCapture(driving_video_path)
        frames = []
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            frames.append(frame)
        cap.release()
        
        # Load pose sequence
        pose_sequence = np.load(pose_sequence_path)
        
        # Apply transformations
        if self.transform:
            reference_image = self.transform(image=reference_image)['image']
            frames = [self.transform(image=frame)['image'] for frame in frames]
        
        # Convert to tensors
        reference_image = torch.from_numpy(reference_image).permute(2, 0, 1).float()
        driving_video = torch.stack([torch.from_numpy(frame).permute(2, 0, 1).float() for frame in frames])
        pose_sequence = torch.from_numpy(pose_sequence).float()
        
        return reference_image, driving_video, pose_sequence, text_prompt
