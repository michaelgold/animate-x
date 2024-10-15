import torch
import torch.nn as nn
import pytorch_lightning as pl
from diffusers import AutoencoderKL, UNet2DConditionModel
from transformers import CLIPTextModel, CLIPTokenizer, CLIPVisionModel
from mamba_ssm import Mamba
from models import ImplicitPoseIndicator, ExplicitPoseIndicator
from torchmetrics.image.fid import FrechetInceptionDistance
from torchmetrics.image.ssim import StructuralSimilarityIndexMeasure
from torchmetrics.image.lpip import LearnedPerceptualImagePatchSimilarity

class AnimateX(pl.LightningModule):
    def __init__(self, config):
        super().__init__()
        self.config = config
        
        # Latent Diffusion Model components
        self.vae = AutoencoderKL.from_pretrained(config['vae_path'])
        self.unet = UNet2DConditionModel.from_pretrained(config['unet_path'])
        
        # CLIP for text and image conditioning
        self.text_encoder = CLIPTextModel.from_pretrained(config['clip_path'])
        self.vision_encoder = CLIPVisionModel.from_pretrained(config['clip_path'])
        self.tokenizer = CLIPTokenizer.from_pretrained(config['clip_path'])
        
        # Mamba for temporal modeling
        self.mamba = Mamba(
            d_model=config['mamba_dim'],
            d_state=config['mamba_state_dim'],
            d_conv=config['mamba_conv_dim'],
            expand=config['mamba_expand']
        )
        
        # Implicit Pose Indicator (IPI)
        self.ipi = self._build_ipi()
        
        # Explicit Pose Indicator (EPI)
        self.epi = ExplicitPoseIndicator(config)
        
        # Loss functions
        self.l1_loss = nn.L1Loss()
        self.mse_loss = nn.MSELoss()
        
        # Configure metrics
        self.configure_metrics()
    
    def _build_ipi(self):
        return ImplicitPoseIndicator(self.config)
    
    def _process_driving_video(self, driving_video):
        batch_size, num_frames, channels, height, width = driving_video.shape
        driving_video = driving_video.view(-1, channels, height, width)
        clip_features = self.vision_encoder(driving_video).last_hidden_state[:, 0]
        clip_features = clip_features.view(batch_size, num_frames, -1)
        
        # Apply Mamba for temporal modeling
        mamba_output = self.mamba(clip_features)
        
        return mamba_output, clip_features
    
    def forward(self, source_image, driving_video, text_prompt, pose_sequence):
        assert source_image.shape[1:] == (3, 256, 256), f"Expected source_image shape (3, 256, 256), got {source_image.shape[1:]}"
        assert driving_video.shape[2:] == (3, 256, 256), f"Expected driving_video shape (B, T, 3, 256, 256), got {driving_video.shape[2:]}"
        assert pose_sequence.shape[-1] == self.config['pose_dim'], f"Expected pose_sequence dim {self.config['pose_dim']}, got {pose_sequence.shape[-1]}"
        
        source_latent = self.vae.encode(source_image).latent_dist.sample()
        driving_features, clip_features = self._process_driving_video(driving_video)
        
        # Generate implicit pose information
        implicit_pose_info = self.ipi(clip_features, pose_sequence)
        explicit_pose_info = self.epi(pose_sequence)
        
        # Combine pose information
        pose_info = torch.cat([implicit_pose_info, explicit_pose_info], dim=-1)
        
        # Encode the text prompt
        text_embeddings = self._encode_text(text_prompt)
        
        latent_model_input = torch.cat([source_latent.repeat(driving_features.shape[1], 1, 1, 1), pose_info, driving_features], dim=1)
        noise_pred = self.unet(latent_model_input, text_embeddings).sample
        
        animation = self.vae.decode(noise_pred).sample
        
        return animation
    
    def _encode_text(self, text_prompt):
        text_input = self.tokenizer(text_prompt, padding="max_length", max_length=self.tokenizer.model_max_length, truncation=True, return_tensors="pt")
        return self.text_encoder(text_input.input_ids)[0]
    
    def training_step(self, batch, batch_idx):
        source_image, driving_video, text_prompt, pose_sequence, target_animation = batch
        
        generated_animation = self(source_image, driving_video, text_prompt, pose_sequence)
        
        reconstruction_loss = self.l1_loss(generated_animation, target_animation)
        perceptual_loss = self.mse_loss(self.vision_encoder(generated_animation).last_hidden_state,
                                        self.vision_encoder(target_animation).last_hidden_state)
        temporal_loss = self.temporal_consistency_loss(generated_animation)
        
        total_loss = (
            self.config['reconstruction_weight'] * reconstruction_loss +
            self.config['perceptual_weight'] * perceptual_loss +
            self.config['temporal_weight'] * temporal_loss
        )
        
        self.log('train_loss', total_loss)
        self.log('train_reconstruction_loss', reconstruction_loss)
        self.log('train_perceptual_loss', perceptual_loss)
        self.log('train_temporal_loss', temporal_loss)
        
        return total_loss
    
    def validation_step(self, batch, batch_idx):
        source_image, driving_video, text_prompt, pose_sequence, target_animation = batch
        
        generated_animation = self(source_image, driving_video, text_prompt, pose_sequence)
        
        reconstruction_loss = self.l1_loss(generated_animation, target_animation)
        perceptual_loss = self.mse_loss(self.vision_encoder(generated_animation).last_hidden_state,
                                        self.vision_encoder(target_animation).last_hidden_state)
        temporal_loss = self.temporal_consistency_loss(generated_animation)
        
        total_loss = (
            self.config['reconstruction_weight'] * reconstruction_loss +
            self.config['perceptual_weight'] * perceptual_loss +
            self.config['temporal_weight'] * temporal_loss
        )
        
        self.log('val_loss', total_loss)
        self.log('val_reconstruction_loss', reconstruction_loss)
        self.log('val_perceptual_loss', perceptual_loss)
        self.log('val_temporal_loss', temporal_loss)
        
        # Update metrics
        self.fid.update(generated_animation, real=False)
        self.fid.update(target_animation, real=True)
        self.ssim.update(generated_animation, target_animation)
        self.lpips.update(generated_animation, target_animation)
        
        return total_loss
    
    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=self.config['learning_rate'])
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=5)
        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": scheduler,
                "monitor": "val_loss",
            },
        }

    def temporal_consistency_loss(self, generated_animation):
        diff = generated_animation[:, 1:] - generated_animation[:, :-1]
        return torch.mean(torch.abs(diff))

    def configure_metrics(self):
        self.fid = FrechetInceptionDistance(feature=2048)
        self.ssim = StructuralSimilarityIndexMeasure()
        self.lpips = LearnedPerceptualImagePatchSimilarity(net_type='alex')

    def on_validation_epoch_end(self):
        fid_score = self.fid.compute()
        ssim_score = self.ssim.compute()
        lpips_score = self.lpips.compute()
        
        self.log('val_fid', fid_score)
        self.log('val_ssim', ssim_score)
        self.log('val_lpips', lpips_score)
        
        # Reset metrics for next epoch
        self.fid.reset()
        self.ssim.reset()
        self.lpips.reset()
