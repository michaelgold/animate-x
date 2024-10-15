import torch
from torch.utils.data import Dataset
import cv2
import numpy as np
import os
import albumentations as A
from albumentations.pytorch import ToTensorV2
from transformers import CLIPTokenizer, AutoFeatureExtractor, AutoModel

class AnimateXDataset(Dataset):
    """
    Dataset class for the AnimateX model.
    
    This dataset loads and preprocesses the data required for training and evaluating
    the AnimateX model, including reference images, driving videos, pose sequences,
    and text prompts.
    """

    def __init__(self, data_dir, config, transform=None, use_cache=True):
        """
        Initialize the AnimateXDataset.
        
        Args:
            data_dir (str): Path to the directory containing the dataset.
            config (dict): Configuration dictionary containing necessary parameters.
            transform (callable, optional): A function/transform to apply to the image data.
                If None, a default transform is applied.
            use_cache (bool, optional): Whether to use caching for faster data loading.
                Defaults to True.
        """
        self.data_dir = data_dir
        self.config = config
        self.transform = transform or self._default_transform()
        self.use_cache = use_cache
        self.tokenizer = CLIPTokenizer.from_pretrained(config['clip_path'])
        self.samples = self._load_samples()
        self.cache = {}
        self.dwpose_extractor = AutoFeatureExtractor.from_pretrained(config['model']['dwpose_path'])
        self.dwpose_model = AutoModel.from_pretrained(config['model']['dwpose_path'])
    
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
        video_path = self.samples[idx]
        
        # Load video
        cap = cv2.VideoCapture(video_path)
        frames = []
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            frames.append(frame)
        cap.release()
        
        # Use first frame as reference image
        reference_image = frames[0]
        
        # Extract pose sequence using DWPose
        inputs = self.dwpose_extractor(images=frames, return_tensors="pt")
        with torch.no_grad():
            outputs = self.dwpose_model(**inputs)
        pose_sequence = outputs.last_hidden_state.squeeze(0).numpy()
        
        # Generate or load text prompt (you'll need to implement this part)
        text_prompt = self._generate_text_prompt(video_path)
        
        # Apply transformations
        if self.transform:
            reference_image = self.transform(image=reference_image)['image']
            frames = [self.transform(image=frame)['image'] for frame in frames]
        
        # Convert to tensors
        reference_image = torch.from_numpy(reference_image).permute(2, 0, 1).float()
        driving_video = torch.stack([torch.from_numpy(frame).permute(2, 0, 1).float() for frame in frames])
        pose_sequence = torch.from_numpy(pose_sequence).float()
        
        return reference_image, driving_video, pose_sequence, text_prompt
