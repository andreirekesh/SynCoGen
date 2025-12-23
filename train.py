#!/usr/bin/env python
"""Training script for SynCoGen diffusion model.

Usage:
    python -m syncogen.train --config syncogen/configs/experiments/default.gin
    python -m syncogen.train --config syncogen/configs/experiments/overfit.gin
    
With overrides:
    python -m syncogen.train --config syncogen/configs/experiments/default.gin \
        --gin "Optimizer.lr=1e-3" \
        --gin "Trainer.max_steps=50000"
        
Resume from checkpoint:
    python -m syncogen.train --config syncogen/configs/experiments/default.gin \
        --ckpt_path checkpoints/last.ckpt
"""

import argparse
from datetime import datetime
from pathlib import Path

import gin
import lightning as L
from syncogen.constants.constants import load_vocabulary


def get_timestamp() -> str:
    """Get a timestamp string for run naming."""
    return datetime.now().strftime("%Y%m%d_%H%M%S")


def parse_args():
    parser = argparse.ArgumentParser(description="Train SynCoGen diffusion model")
    parser.add_argument(
        "--config",
        type=str,
        required=True,
        help="Path to gin config file (e.g., syncogen/configs/experiments/default.gin)",
    )
    parser.add_argument(
        "--vocab_dir", type=str, required=True, help="Path to the vocabulary directory"
    )
    parser.add_argument(
        "--gin",
        type=str,
        action="append",
        default=[],
        help="Additional gin bindings (can be specified multiple times)",
    )
    parser.add_argument(
        "--ckpt_path", type=str, default=None, help="Path to checkpoint to resume from"
    )
    parser.add_argument("--seed", type=int, default=42, help="Random seed for reproducibility")
    return parser.parse_args()


### TODO: Hacky, I have to import here otherwise constants will not be populated.
args = parse_args()
load_vocabulary(args.vocab_dir)

from syncogen.diffusion.training.trainer import Trainer
from syncogen.diffusion.training.diffusion import Diffusion
from syncogen.data.dataloader import SyncogenDataManager
import syncogen.logging.loggers
import syncogen.logging.callbacks


@gin.configurable
def train(
    seed: int = 42,
    ckpt_path: str = None,
    vocab_dir: str = None,
):
    """Main training function.

    Args:
        seed: Random seed for reproducibility
        ckpt_path: Path to checkpoint to resume from
        vocab_dir: Path to the vocabulary directory
    """
    # Initialize the constants
    load_vocabulary(vocab_dir)
    # Set seed for reproducibility
    L.seed_everything(seed, workers=True)

    # Create data manager and get dataloaders
    data_manager = SyncogenDataManager()
    train_loader, val_loader = data_manager.get_dataloaders()

    # Create model (gin-configured)
    model = Diffusion(data_manager=data_manager)

    trainer_wrapper = Trainer()
    trainer = trainer_wrapper.trainer

    # Log gin config to wandb (only on rank 0)
    if trainer.logger is not None and trainer.global_rank == 0:
        try:
            trainer.logger.experiment.config.update({"gin_config": gin.operative_config_str()})
        except (AttributeError, TypeError):
            pass  # Logger not fully initialized yet, will be logged later

    # Train!
    trainer.fit(
        model=model,
        train_dataloaders=train_loader,
        val_dataloaders=val_loader,
        ckpt_path=ckpt_path,
    )

    return trainer, model


def main():
    # Generate run name from config file name and timestamp
    config_name = Path(args.config).stem

    # Parse gin config file and any additional bindings
    gin_bindings = args.gin
    gin.parse_config_files_and_bindings(
        config_files=[args.config],
        bindings=gin_bindings,
    )

    # Run training
    trainer, model = train(
        seed=args.seed,
        ckpt_path=args.ckpt_path,
        vocab_dir=args.vocab_dir,
    )

    # Print final summary
    print(f"\n{'='*60}")
    print(f"Training completed!")
    print(
        f"Best model checkpoint: {trainer.checkpoint_callback.best_model_path if trainer.checkpoint_callback else 'N/A'}"
    )
    print(f"{'='*60}\n")

    # Print operative gin config for reproducibility
    print("Operative Gin Config:")
    print(gin.operative_config_str())


if __name__ == "__main__":
    main()
