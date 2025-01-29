import torch
import copy

class ModelEMA:
    """
    Exponential Moving Average for PyTorch models.
    Call update() after each step to update shadow weights.
    Call apply_shadow() to temporarily replace model weights for testing/saving.
    """
    def __init__(self, model, decay=0.9999):
        self.decay = decay
        # Create shadow copy
        self.ema_model = self._clone_model(model)
    
    def _clone_model(self, model):
        ema_model = copy.deepcopy(model)  # Use deepcopy instead
        for p in ema_model.parameters():
            p.requires_grad_(False)
        return ema_model

    def update(self, model):
        # Update shadow weights
        with torch.no_grad():
            ema_sd = self.ema_model.state_dict()
            mod_sd = model.state_dict()
            for k in ema_sd.keys():
                if ema_sd[k].dtype.is_floating_point:
                    ema_sd[k].data.mul_(self.decay).add_((1. - self.decay)*mod_sd[k].data)
                else:
                    ema_sd[k] = mod_sd[k]

    def apply_shadow(self, model):
        # Copy shadow weights back to model (for testing/saving)
        model.load_state_dict(self.ema_model.state_dict())
