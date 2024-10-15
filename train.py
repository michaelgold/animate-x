import pytorch_lightning as pl
from pytorch_lightning.loggers import TensorBoardLogger
from pytorch_lightning.callbacks import ModelCheckpoint, LearningRateMonitor
from torch.utils.data import DataLoader
import yaml
import torch

from models.animate_x import AnimateX
from data.dataset import AnimateXDataset

def validate_config(config):
    required_keys = ['data', 'model', 'training']
    for key in required_keys:
        if key not in config:
            raise ValueError(f"Missing required key '{key}' in config file")

def main():
    # Load and validate configuration
    with open('config/config.yaml', 'r') as f:
        config = yaml.safe_load(f)
    validate_config(config)
    
    # Create datasets and data loaders
    train_dataset = AnimateXDataset(config['data']['data_dir'], split='train')
    val_dataset = AnimateXDataset(config['data']['data_dir'], split='val')
    
    train_loader = DataLoader(
        train_dataset,
        batch_size=config['data']['batch_size'],
        num_workers=config['data']['num_workers'],
        shuffle=True
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=config['data']['batch_size'],
        num_workers=config['data']['num_workers'],
        shuffle=False
    )
    
    # Create model
    model = AnimateX(config['model'])
    
    # Create logger
    logger = TensorBoardLogger("logs", name="animate_x")
    
    # Create callbacks
    checkpoint_callback = ModelCheckpoint(
        dirpath='checkpoints',
        filename='animate_x-{epoch:02d}-{val_loss:.2f}',
        save_top_k=3,
        monitor='val_loss',
        mode='min'
    )
    
    lr_monitor = LearningRateMonitor(logging_interval='step')
    
    # Add learning rate scheduler
    lr_scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode='min',
        factor=config['training']['lr_factor'],
        patience=config['training']['lr_patience'],
        verbose=True
    )
    
    lr_scheduler_callback = pl.callbacks.LearningRateMonitor(logging_interval='epoch')
    
    # Create trainer
    trainer = pl.Trainer(
        max_epochs=config['training']['max_epochs'],
        logger=logger,
        callbacks=[checkpoint_callback, lr_monitor, lr_scheduler_callback],
        gpus=1 if torch.cuda.is_available() else 0,
        gradient_clip_val=config['training']['clip_grad_norm']
    )
    
    # Train the model
    trainer.fit(model, train_loader, val_loader)

if __name__ == "__main__":
    main()
