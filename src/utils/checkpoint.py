"""
Checkpoint management for bijective discrete diffusion models.
Handles saving, loading, and resuming training with full state preservation.
"""

import torch
import os
import json
import shutil
from datetime import datetime
from typing import Dict, Any, Optional, Tuple
from pathlib import Path


class CheckpointManager:
    """
    Comprehensive checkpoint management for bijective diffusion training.
    
    Features:
    - Automatic checkpoint saving during training
    - Resume training from any checkpoint
    - Best model tracking based on validation metrics
    - Model export for inference deployment
    - Training history preservation
    """
    
    def __init__(
        self,
        checkpoint_dir: str = "models/checkpoints",
        export_dir: str = "models/exports",
        max_checkpoints: int = 3,
        save_every_n_epochs: int = 2
    ):
        self.checkpoint_dir = Path(checkpoint_dir)
        self.export_dir = Path(export_dir)
        self.max_checkpoints = max_checkpoints
        self.save_every_n_epochs = save_every_n_epochs
        
        # Create directories
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        self.export_dir.mkdir(parents=True, exist_ok=True)
        
        # Training history
        self.history_file = self.checkpoint_dir / "training_history.json"
        self.training_history = self._load_history()
        
        # Best model tracking
        self.best_loss = float('inf')
        self.best_model_path = self.checkpoint_dir / "best_model.pt"
    
    def _load_history(self) -> Dict[str, Any]:
        """Load training history from file."""
        if self.history_file.exists():
            with open(self.history_file, 'r') as f:
                return json.load(f)
        return {
            "epochs": [],
            "losses": [],
            "validation_losses": [],
            "learning_rates": [],
            "timestamps": [],
            "model_info": {}
        }
    
    def _save_history(self):
        """Save training history to file."""
        with open(self.history_file, 'w') as f:
            json.dump(self.training_history, f, indent=2, default=str)
    
    def save_checkpoint(
        self,
        model: torch.nn.Module,
        optimizer: torch.optim.Optimizer,
        scheduler: Optional[torch.optim.lr_scheduler._LRScheduler],
        epoch: int,
        loss: float,
        config: Any,
        validation_loss: Optional[float] = None,
        additional_data: Optional[Dict[str, Any]] = None
    ) -> str:
        """
        Save a comprehensive checkpoint.
        
        Args:
            model: The bijective diffusion model
            optimizer: Training optimizer
            scheduler: Learning rate scheduler (optional)
            epoch: Current epoch number
            loss: Training loss
            config: Model configuration
            validation_loss: Validation loss (optional)
            additional_data: Any additional data to save
            
        Returns:
            Path to saved checkpoint
        """
        # Create checkpoint data
        checkpoint = {
            'epoch': epoch,
            'loss': loss,
            'validation_loss': validation_loss,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'config': config,
            'timestamp': datetime.now().isoformat(),
            'learning_rate': optimizer.param_groups[0]['lr']
        }
        
        # Add scheduler state if available
        if scheduler is not None:
            checkpoint['scheduler_state_dict'] = scheduler.state_dict()
        
        # Add model info
        if hasattr(model, 'get_bijective_info'):
            checkpoint['bijective_info'] = model.get_bijective_info()
        elif hasattr(model, 'module') and hasattr(model.module, 'get_bijective_info'):
            checkpoint['bijective_info'] = model.module.get_bijective_info()
        
        # Add additional data
        if additional_data:
            checkpoint.update(additional_data)
        
        # Create checkpoint filename
        checkpoint_name = f"epoch_{epoch:03d}_loss_{loss:.4f}.pt"
        checkpoint_path = self.checkpoint_dir / checkpoint_name
        
        # Save checkpoint
        torch.save(checkpoint, checkpoint_path)
        print(f"✅ Checkpoint saved: {checkpoint_path}")
        
        # Update training history
        self.training_history["epochs"].append(epoch)
        self.training_history["losses"].append(loss)
        self.training_history["validation_losses"].append(validation_loss)
        self.training_history["learning_rates"].append(optimizer.param_groups[0]['lr'])
        self.training_history["timestamps"].append(datetime.now().isoformat())
        
        # Save model info (first time only)
        if not self.training_history["model_info"] and 'bijective_info' in checkpoint:
            self.training_history["model_info"] = checkpoint['bijective_info']
        
        self._save_history()
        
        # Check if this is the best model
        if validation_loss is not None and validation_loss < self.best_loss:
            self.best_loss = validation_loss
            shutil.copy2(checkpoint_path, self.best_model_path)
            print(f"🏆 New best model saved: {self.best_model_path}")
        
        # Clean up old checkpoints
        self._cleanup_old_checkpoints()
        
        return str(checkpoint_path)
    
    def load_checkpoint(
        self,
        model: torch.nn.Module,
        optimizer: torch.optim.Optimizer,
        scheduler: Optional[torch.optim.lr_scheduler._LRScheduler],
        checkpoint_path: str,
        device: str = "cpu"
    ) -> Tuple[int, float, Any]:
        """
        Load a checkpoint and restore training state.
        
        Args:
            model: The bijective diffusion model
            optimizer: Training optimizer
            scheduler: Learning rate scheduler (optional)
            checkpoint_path: Path to checkpoint file
            device: Device to load checkpoint on
            
        Returns:
            Tuple of (epoch, loss, config)
        """
        if not os.path.exists(checkpoint_path):
            raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")
        
        print(f"📂 Loading checkpoint: {checkpoint_path}")
        
        # Load checkpoint (use weights_only=False for our trusted checkpoints)
        checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
        
        # Restore model state
        model.load_state_dict(checkpoint['model_state_dict'])
        
        # Restore optimizer state
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        
        # Restore scheduler state if available
        if scheduler is not None and 'scheduler_state_dict' in checkpoint:
            scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        
        epoch = checkpoint['epoch']
        loss = checkpoint['loss']
        config = checkpoint['config']
        
        print(f"✅ Checkpoint loaded: Epoch {epoch}, Loss {loss:.4f}")
        
        # Print model info if available
        if 'bijective_info' in checkpoint:
            info = checkpoint['bijective_info']
            print(f"   Model: {info['total_params']:,} parameters")
            print(f"   Type: {info['model_type']}")
        
        return epoch, loss, config
    
    def export_model(
        self,
        model: torch.nn.Module,
        config: Any,
        export_name: str = "bijective_diffusion_model",
        include_optimizer: bool = False
    ) -> str:
        """
        Export model for inference deployment.
        
        Args:
            model: The trained bijective diffusion model
            config: Model configuration
            export_name: Name for exported model
            include_optimizer: Whether to include optimizer state
            
        Returns:
            Path to exported model
        """
        export_data = {
            'model_state_dict': model.state_dict(),
            'config': config,
            'export_timestamp': datetime.now().isoformat(),
            'export_type': 'inference'
        }
        
        # Add model info
        if hasattr(model, 'get_bijective_info'):
            export_data['bijective_info'] = model.get_bijective_info()
        elif hasattr(model, 'module') and hasattr(model.module, 'get_bijective_info'):
            export_data['bijective_info'] = model.module.get_bijective_info()
        
        # Add training history summary
        if self.training_history["epochs"]:
            export_data['training_summary'] = {
                'total_epochs': max(self.training_history["epochs"]),
                'final_loss': self.training_history["losses"][-1],
                'best_validation_loss': min([l for l in self.training_history["validation_losses"] if l is not None], default=None)
            }
        
        # Create export path
        export_path = self.export_dir / f"{export_name}.pt"
        
        # Save export
        torch.save(export_data, export_path)
        print(f"📦 Model exported: {export_path}")
        
        # Also save config as JSON for easy inspection
        config_path = self.export_dir / f"{export_name}_config.json"
        with open(config_path, 'w') as f:
            json.dump(export_data['bijective_info'], f, indent=2, default=str)
        
        return str(export_path)
    
    def load_exported_model(
        self,
        model: torch.nn.Module,
        export_path: str,
        device: str = "cpu"
    ) -> Any:
        """
        Load an exported model for inference.
        
        Args:
            model: Empty model instance to load weights into
            export_path: Path to exported model
            device: Device to load model on
            
        Returns:
            Model configuration
        """
        if not os.path.exists(export_path):
            raise FileNotFoundError(f"Exported model not found: {export_path}")
        
        print(f"📦 Loading exported model: {export_path}")
        
        # Load export data (use weights_only=False for our trusted exports)
        export_data = torch.load(export_path, map_location=device, weights_only=False)
        
        # Restore model state
        model.load_state_dict(export_data['model_state_dict'])
        
        config = export_data['config']
        
        print(f"✅ Exported model loaded")
        
        # Print model info
        if 'bijective_info' in export_data:
            info = export_data['bijective_info']
            print(f"   Model: {info['total_params']:,} parameters")
            print(f"   Type: {info['model_type']}")
        
        if 'training_summary' in export_data:
            summary = export_data['training_summary']
            print(f"   Training: {summary['total_epochs']} epochs")
            print(f"   Final loss: {summary['final_loss']:.4f}")
        
        return config
    
    def _cleanup_old_checkpoints(self):
        """Remove old checkpoints, keeping only the most recent ones."""
        # Get all checkpoint files (excluding best_model.pt)
        checkpoint_files = [
            f for f in self.checkpoint_dir.glob("epoch_*.pt")
            if f.name != "best_model.pt"
        ]
        
        # Sort by modification time (newest first)
        checkpoint_files.sort(key=lambda x: x.stat().st_mtime, reverse=True)
        
        # Remove old checkpoints
        for old_checkpoint in checkpoint_files[self.max_checkpoints:]:
            old_checkpoint.unlink()
            print(f"🗑️  Removed old checkpoint: {old_checkpoint.name}")
    
    def list_checkpoints(self) -> list:
        """List all available checkpoints."""
        checkpoints = []
        
        for checkpoint_file in self.checkpoint_dir.glob("epoch_*.pt"):
            if checkpoint_file.name == "best_model.pt":
                continue
                
            try:
                # Load checkpoint metadata (use weights_only=False for trusted checkpoints)
                checkpoint = torch.load(checkpoint_file, map_location='cpu', weights_only=False)
                checkpoints.append({
                    'path': str(checkpoint_file),
                    'epoch': checkpoint.get('epoch', 0),
                    'loss': checkpoint.get('loss', 0.0),
                    'validation_loss': checkpoint.get('validation_loss'),
                    'timestamp': checkpoint.get('timestamp', ''),
                    'size_mb': checkpoint_file.stat().st_size / (1024 * 1024)
                })
            except Exception as e:
                print(f"⚠️  Could not read checkpoint {checkpoint_file}: {e}")
        
        # Sort by epoch
        checkpoints.sort(key=lambda x: x['epoch'])
        
        return checkpoints
    
    def get_latest_checkpoint(self) -> Optional[str]:
        """Get path to the most recent checkpoint."""
        checkpoints = self.list_checkpoints()
        if checkpoints:
            return checkpoints[-1]['path']
        return None
    
    def get_best_checkpoint(self) -> Optional[str]:
        """Get path to the best model checkpoint."""
        if self.best_model_path.exists():
            return str(self.best_model_path)
        return None
    
    def should_save_checkpoint(self, epoch: int) -> bool:
        """Check if checkpoint should be saved for this epoch."""
        return epoch % self.save_every_n_epochs == 0
    
    def get_training_summary(self) -> Dict[str, Any]:
        """Get summary of training progress."""
        if not self.training_history["epochs"]:
            return {"status": "No training history"}
        
        return {
            "total_epochs": max(self.training_history["epochs"]),
            "total_checkpoints": len(self.training_history["epochs"]),
            "best_loss": min(self.training_history["losses"]),
            "latest_loss": self.training_history["losses"][-1],
            "best_validation_loss": min([l for l in self.training_history["validation_losses"] if l is not None], default=None),
            "model_info": self.training_history["model_info"],
            "training_duration": len(self.training_history["epochs"]) * self.save_every_n_epochs
        }


def create_checkpoint_manager(
    checkpoint_dir: str = "models/checkpoints",
    export_dir: str = "models/exports",
    max_checkpoints: int = 3,
    save_every_n_epochs: int = 2
) -> CheckpointManager:
    """Create a checkpoint manager with default settings."""
    return CheckpointManager(
        checkpoint_dir=checkpoint_dir,
        export_dir=export_dir,
        max_checkpoints=max_checkpoints,
        save_every_n_epochs=save_every_n_epochs
    )
