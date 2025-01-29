"""Metrics tracking utilities for training."""
import time
from typing import Dict, Optional

import numpy as np
import torch
from torch.utils.tensorboard import SummaryWriter


class AverageMeter:
    """Computes and stores the average and current value."""
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


class MetricLogger:
    """Tracks training metrics and handles logging."""
    def __init__(
        self,
        log_dir: str,
        experiment_name: str,
        use_tensorboard: bool = True,
        log_interval: int = 10
    ):
        self.metrics = {}
        self.log_interval = log_interval
        self.start_time = time.time()
        self.log_file = f"{log_dir}/{experiment_name}.log"
        
        # Initialize tensorboard
        self.use_tensorboard = use_tensorboard
        if use_tensorboard:
            self.writer = SummaryWriter(f"{log_dir}/tensorboard/{experiment_name}")
        
        # Create metric trackers
        self.create_metric("train_loss")
        self.create_metric("cls_loss")
        self.create_metric("adv_loss")
        self.create_metric("sparse_loss")
        self.create_metric("target_acc")
    
    def create_metric(self, name: str):
        """Creates a new metric tracker."""
        self.metrics[name] = AverageMeter()
    
    def update(self, name: str, value: float, n: int = 1):
        """Updates a metric value."""
        if name not in self.metrics:
            self.create_metric(name)
        self.metrics[name].update(value, n)
    
    def get_value(self, name: str) -> float:
        """Gets the current value of a metric."""
        return self.metrics[name].avg if name in self.metrics else 0.0
    
    def log_epoch(
        self,
        epoch: int,
        total_epochs: int,
        learning_rate: float,
        additional_info: Optional[Dict] = None
    ):
        """Logs metrics for the current epoch."""
        if epoch % self.log_interval != 0 and epoch != total_epochs:
            return
        
        # Calculate elapsed time
        elapsed = time.time() - self.start_time
        
        # Format the log message
        log_msg = [
            f"Epoch [{epoch}/{total_epochs}] | Time: {int(elapsed)}s",
            "   - Training Loss: {:.3f}".format(self.get_value("train_loss")),
            "       - Classification Loss: {:.3f}".format(self.get_value("cls_loss")),
            "       - Domain Adv Loss: {:.3f}".format(self.get_value("adv_loss")),
            "       - Sparsity Reg Loss: {:.3f}".format(self.get_value("sparse_loss")),
            "   - Target Accuracy: {:.2f}%".format(self.get_value("target_acc")),
            "   - Current LR: {:.4f}".format(learning_rate)
        ]
        
        # Add any additional info
        if additional_info:
            for key, value in additional_info.items():
                log_msg.append(f"   - {key}: {value}")
        
        # Write to log file
        with open(self.log_file, "a") as f:
            f.write("\n".join(log_msg) + "\n\n")
        
        # Log to tensorboard
        if self.use_tensorboard:
            self.writer.add_scalar("Loss/Total", self.get_value("train_loss"), epoch)
            self.writer.add_scalar("Loss/Classification", self.get_value("cls_loss"), epoch)
            self.writer.add_scalar("Loss/Adversarial", self.get_value("adv_loss"), epoch)
            self.writer.add_scalar("Loss/Sparsity", self.get_value("sparse_loss"), epoch)
            self.writer.add_scalar("Accuracy/Target", self.get_value("target_acc"), epoch)
            self.writer.add_scalar("Learning_Rate", learning_rate, epoch)
    
    def reset(self):
        """Resets all metric trackers."""
        for metric in self.metrics.values():
            metric.reset()
    
    def close(self):
        """Closes the tensorboard writer."""
        if self.use_tensorboard:
            self.writer.close()
