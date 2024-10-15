import torch
import torch.nn as nn
import torch.nn.functional as F

class ImplicitPoseIndicator(nn.Module):
    """
    Implicit Pose Indicator (IPI) module for the AnimateX model.
    
    This module combines CLIP features and pose sequence information to generate
    implicit pose representations. It uses a multi-layer perceptron (MLP) to process
    the combined features and produce the implicit pose representation.
    """

    def __init__(self, config):
        """
        Initialize the Implicit Pose Indicator.

        Args:
            config (dict): Configuration dictionary containing model parameters.
                Expected keys:
                - clip_feature_dim (int): Dimension of the CLIP features.
                - hidden_dim (int): Dimension of the hidden layers in the MLP.
                - ipi_output_dim (int): Dimension of the output implicit pose representation.
        """
        super().__init__()
        self.feature_dim = config['clip_feature_dim']
        self.hidden_dim = config['hidden_dim']
        self.output_dim = config['ipi_output_dim']
        
        # Multi-layer perceptron for processing combined features
        self.mlp = nn.Sequential(
            nn.Linear(self.feature_dim, self.hidden_dim),
            nn.ReLU(),
            nn.Linear(self.hidden_dim, self.hidden_dim),
            nn.ReLU(),
            nn.Linear(self.hidden_dim, self.output_dim)
        )
    
    def forward(self, clip_features, pose_sequence):
        """
        Forward pass of the Implicit Pose Indicator.

        This method combines the CLIP features and pose sequence, then processes
        them through the MLP to generate the implicit pose representation.

        Args:
            clip_features (torch.Tensor): CLIP features extracted from the driving video.
                Shape: (batch_size, sequence_length, clip_feature_dim)
            pose_sequence (torch.Tensor): Pose sequence information.
                Shape: (batch_size, sequence_length, pose_dim)

        Returns:
            torch.Tensor: Implicit pose representation.
                Shape: (batch_size, sequence_length, ipi_output_dim)
        """
        # Combine CLIP features and pose sequence
        combined_features = torch.cat([clip_features, pose_sequence], dim=-1)
        return self.mlp(combined_features)

class ExplicitPoseIndicator(nn.Module):
    """
    Explicit Pose Indicator (EPI) module for the AnimateX model.
    
    This module processes pose sequences to generate explicit pose representations
    and simulates pose transformations. It uses a combination of convolutional
    layers for encoding, a pose pool for similarity matching, and an MLP for
    pose transformation.
    """

    def __init__(self, config):
        """
        Initialize the Explicit Pose Indicator.

        Args:
            config (dict): Configuration dictionary containing model parameters.
                Expected keys:
                - pose_dim (int): Dimension of the input pose sequence.
                - hidden_dim (int): Dimension of the hidden layers.
                - epi_output_dim (int): Dimension of the output explicit pose representation.
                - num_pose_anchors (int): Number of pose anchors in the pose pool.
        """
        super().__init__()
        self.pose_encoder = nn.Sequential(
            nn.Conv1d(config.pose_dim, config.hidden_dim, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv1d(config.hidden_dim, config.epi_output_dim, kernel_size=3, padding=1)
        )
        self.pose_pool = nn.Parameter(torch.randn(config.num_pose_anchors, config.pose_dim))
        self.transformation_mlp = nn.Sequential(
            nn.Linear(config.pose_dim * 2, config.hidden_dim),
            nn.ReLU(),
            nn.Linear(config.hidden_dim, config.pose_dim)
        )
        
    def forward(self, pose_sequence):
        """
        Forward pass of the Explicit Pose Indicator.

        This method encodes the input pose sequence and combines it with a
        transformed pose representation to generate the final explicit pose representation.

        Args:
            pose_sequence (torch.Tensor): Input pose sequence.
                Shape: (batch_size, sequence_length, pose_dim)

        Returns:
            torch.Tensor: Explicit
        """
        # Encode pose sequence
        encoded_pose = self.pose_encoder(pose_sequence.transpose(1, 2)).transpose(1, 2)
        
        # Simulate pose transformation
        transformed_pose = self.transform_pose(pose_sequence)
        
        return encoded_pose + transformed_pose
    
    def transform_pose(self, pose_sequence):
        similarity = F.cosine_similarity(pose_sequence.unsqueeze(2), self.pose_pool.unsqueeze(0).unsqueeze(0), dim=-1)
        weights = F.softmax(similarity, dim=-1)
        
        # Select top-k similar poses
        k = 5
        top_k_weights, top_k_indices = torch.topk(weights, k, dim=-1)
        top_k_poses = self.pose_pool[top_k_indices]
        
        # Compute weighted average of top-k poses
        weighted_poses = (top_k_poses * top_k_weights.unsqueeze(-1)).sum(dim=-2)
        
        # Apply transformation MLP
        transformation_input = torch.cat([pose_sequence, weighted_poses], dim=-1)
        transformed = self.transformation_mlp(transformation_input)
        
        return transformed
