import torch

class SAMOptimizer(torch.optim.Optimizer):
    """
    Implementation of Sharpness-Aware Minimization (SAM).
    This version is compatible with AMP and requires first_step()/second_step() usage.
    """
    def __init__(self, base_optimizer, rho=0.05):
        """
        Args:
            base_optimizer: e.g., torch.optim.SGD(...) or AdamW(...)
            rho: perturbation radius
        """
        if not isinstance(base_optimizer, torch.optim.Optimizer):
            raise TypeError("base_optimizer must be torch.optim.Optimizer.")
        defaults = {}
        super().__init__(base_optimizer.param_groups, defaults)
        self.base_optimizer = base_optimizer
        self.rho = rho
        self.param_groups = self.base_optimizer.param_groups

    @torch.no_grad()
    def first_step(self, zero_grad=False):
        """
        After first backward(), call this to add epsilon perturbation to parameters
        """
        grad_norm = self._grad_norm()
        scale = self.rho / (grad_norm + 1e-12)

        for group in self.param_groups:
            for p in group['params']:
                if p.grad is None:
                    continue
                # Add perturbation to p
                e = p.grad * scale
                p.add_(e)

        if zero_grad:
            self.base_optimizer.zero_grad()

    @torch.no_grad()
    def second_step(self, zero_grad=False):
        """
        After second backward(), call this to do actual parameter update and remove perturbation
        """
        self.base_optimizer.step()  # Do actual update
        if zero_grad:
            self.base_optimizer.zero_grad()

    def _grad_norm(self):
        # Calculate L2 norm of all parameter gradients
        shared_device = self.param_groups[0]['params'][0].device
        norm = torch.norm(
            torch.stack([
                p.grad.norm(p=2).to(shared_device)
                for group in self.param_groups
                for p in group['params']
                if p.grad is not None
            ]),
            p=2
        )
        return norm

    def zero_grad(self):
        self.base_optimizer.zero_grad()
