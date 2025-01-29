import torch
import torch.nn as nn

class InterBN(nn.Module):
    """
    Interchangeable BatchNorm implementation.
    Performs BN separately on source/target domains and exchanges channels based on gamma thresholds.
    """
    def __init__(self, num_features, threshold=0.5, momentum=0.1, eps=1e-5):
        super().__init__()
        self.num_features = num_features
        self.threshold = threshold
        self.momentum = momentum
        self.eps = eps
        
        # Source domain parameters
        self.register_parameter('gamma_s', nn.Parameter(torch.ones(num_features)))
        self.register_parameter('beta_s', nn.Parameter(torch.zeros(num_features)))
        self.register_buffer('running_mean_s', torch.zeros(num_features))
        self.register_buffer('running_var_s', torch.ones(num_features))
        
        # Target domain parameters
        self.register_parameter('gamma_t', nn.Parameter(torch.ones(num_features)))
        self.register_parameter('beta_t', nn.Parameter(torch.zeros(num_features)))
        self.register_buffer('running_mean_t', torch.zeros(num_features))
        self.register_buffer('running_var_t', torch.ones(num_features))
    
    def _check_input_dim(self, x):
        if x.dim() != 4:
            raise ValueError(f'expected 4D input (got {x.dim()}D input)')
    
    def _normalize(self, x, mean, var, gamma, beta):
        """Applies batch normalization with given parameters."""
        return gamma.view(1, -1, 1, 1) * (x - mean.view(1, -1, 1, 1)) / \
               torch.sqrt(var.view(1, -1, 1, 1) + self.eps) + beta.view(1, -1, 1, 1)
    
    def _update_running_stats(self, x, running_mean, running_var):
        """Updates running statistics for mean and variance."""
        if self.training:
            with torch.no_grad():
                batch_mean = x.mean([0, 2, 3])
                batch_var = x.var([0, 2, 3], unbiased=False)
                
                running_mean.mul_(1 - self.momentum).add_(batch_mean * self.momentum)
                running_var.mul_(1 - self.momentum).add_(batch_var * self.momentum)
        
        return running_mean, running_var
    
    def _get_exchange_mask(self, gamma_s, gamma_t):
        """Determines which channels should be exchanged based on gamma values."""
        with torch.no_grad():
            exchange_s = (gamma_s.abs() < self.threshold)
            exchange_t = (gamma_t.abs() < self.threshold)
            return exchange_s, exchange_t
    
    def forward(self, x_s, x_t):
        self._check_input_dim(x_s)
        self._check_input_dim(x_t)
        
        # Update running statistics
        self.running_mean_s, self.running_var_s = self._update_running_stats(
            x_s, self.running_mean_s, self.running_var_s
        )
        self.running_mean_t, self.running_var_t = self._update_running_stats(
            x_t, self.running_mean_t, self.running_var_t
        )
        
        # Get exchange masks
        exchange_s, exchange_t = self._get_exchange_mask(self.gamma_s, self.gamma_t)
        
        # Initialize output tensors
        out_s = torch.zeros_like(x_s)
        out_t = torch.zeros_like(x_t)
        
        # Non-exchanged channels
        out_s[:, ~exchange_s] = self._normalize(
            x_s[:, ~exchange_s],
            self.running_mean_s[~exchange_s],
            self.running_var_s[~exchange_s],
            self.gamma_s[~exchange_s],
            self.beta_s[~exchange_s]
        )
        out_t[:, ~exchange_t] = self._normalize(
            x_t[:, ~exchange_t],
            self.running_mean_t[~exchange_t],
            self.running_var_t[~exchange_t],
            self.gamma_t[~exchange_t],
            self.beta_t[~exchange_t]
        )
        
        # Exchanged channels
        out_s[:, exchange_s] = self._normalize(
            x_s[:, exchange_s],
            self.running_mean_t[exchange_s],
            self.running_var_t[exchange_s],
            self.gamma_t[exchange_s],
            self.beta_t[exchange_s]
        )
        out_t[:, exchange_t] = self._normalize(
            x_t[:, exchange_t],
            self.running_mean_s[exchange_t],
            self.running_var_s[exchange_t],
            self.gamma_s[exchange_t],
            self.beta_s[exchange_t]
        )
        
        return out_s, out_t
    
    def get_exchange_ratio(self):
        """Returns the ratio of exchanged channels."""
        with torch.no_grad():
            exchange_s = (self.gamma_s.abs() < self.threshold).float().mean()
            exchange_t = (self.gamma_t.abs() < self.threshold).float().mean()
            return (exchange_s + exchange_t) / 2
