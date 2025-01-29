"""Loss functions for domain adaptation."""
import torch
import torch.nn.functional as F


def domain_adversarial_loss(source_domain, target_domain):
    """
    Calculates the domain adversarial loss.
    
    Args:
        source_domain: Source domain predictions from discriminator
        target_domain: Target domain predictions from discriminator
    
    Returns:
        Domain adversarial loss
    """
    source_labels = torch.ones_like(source_domain)
    target_labels = torch.zeros_like(target_domain)
    
    source_loss = F.binary_cross_entropy_with_logits(
        source_domain, source_labels, reduction='mean'
    )
    target_loss = F.binary_cross_entropy_with_logits(
        target_domain, target_labels, reduction='mean'
    )
    
    return (source_loss + target_loss) / 2


def sparsity_regularization(model):
    """
    Calculates L1 sparsity regularization loss for model parameters.
    Focuses on feature extractor's final layer weights and InterBN parameters.
    
    Args:
        model: The feature extractor model
    
    Returns:
        Sparsity regularization loss
    """
    l1_loss = 0.0
    total_params = 0
    
    # Only apply to conv layers and batch norm parameters
    for name, param in model.named_parameters():
        if 'conv' in name or 'bn' in name:
            if 'weight' in name:  # Only regularize weights, not biases
                l1_loss += torch.abs(param).mean()
                total_params += 1
    
    return l1_loss / max(total_params, 1)


def entropy_loss(logits):
    """
    Calculates entropy minimization loss for target domain predictions.
    
    Args:
        logits: Predicted logits from classifier
    
    Returns:
        Entropy loss
    """
    probs = F.softmax(logits, dim=1)
    log_probs = F.log_softmax(logits, dim=1)
    
    return -(probs * log_probs).sum(dim=1).mean()


class ConsistencyLoss:
    """Implements consistency regularization between original and augmented views."""
    
    def __init__(self, temperature=0.1):
        self.temperature = temperature
    
    def __call__(self, view1, view2):
        """
        Args:
            view1: Features from first view
            view2: Features from second view
        
        Returns:
            Consistency loss between views
        """
        # Normalize feature vectors
        view1 = F.normalize(view1, p=2, dim=1)
        view2 = F.normalize(view2, p=2, dim=1)
        
        # Scaled cosine similarity
        logits = torch.matmul(view1, view2.t()) / self.temperature
        
        # Contrastive loss
        labels = torch.arange(logits.size(0), device=logits.device)
        loss = F.cross_entropy(logits, labels) + F.cross_entropy(logits.t(), labels)
        
        return loss / 2
