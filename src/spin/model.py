import torch
from torch import distributed as dist
from torch import nn
from torch.nn import functional as F
from torch.nn.utils.parametrizations import weight_norm


@torch.no_grad()
def sinkhorn_knopp(x: torch.Tensor, epsilon: float, sinkhorn_iters: int = 3) -> torch.Tensor:
    """Apply Sinkhorn-Knopp algorithm to smooth x.

    Adapted from DINOv2: https://github.com/facebookresearch/dinov2/blob/main/dinov2/loss/ibot_patch_loss.py#L62
    """
    Q = torch.exp(x.float() / epsilon).t()  # Q is K-by-B for consistency with notations from our paper
    K, B = Q.shape  # (prototypes, samples)
    if dist.is_initialized():
        dist.all_reduce(B)

    # make the matrix sums to 1
    sum_Q = torch.sum(Q)
    if dist.is_initialized():
        dist.all_reduce(sum_Q)
    Q /= sum_Q

    for _ in range(sinkhorn_iters):
        # normalize each row: total weight per prototype must be 1/K
        sum_of_rows = torch.sum(Q, dim=1, keepdim=True)
        if dist.is_initialized():
            dist.all_reduce(sum_of_rows)
        Q /= sum_of_rows
        Q /= K

        # normalize each column: total weight per sample must be 1/B
        Q /= torch.sum(Q, dim=0, keepdim=True)
        Q /= B

    Q *= B  # the columns must sum to 1 so that Q is an assignment
    return Q.t()


class Spin(nn.Module):
    def __init__(
        self,
        inp_dim: int,
        codebook_dim: int = 256,
        codebook_size: int = 2048,
        *,
        epsilon: float = 0.02,
        temperature: float = 0.1,
        sinkhorn_iters: int = 3,
    ) -> None:
        super().__init__()
        self.head = nn.Linear(inp_dim, codebook_dim)
        self.codebook = weight_norm(nn.Linear(codebook_dim, codebook_size, bias=False))
        self.codebook.parametrizations.weight.original0.data.fill_(1)

        self.epsilon = epsilon
        self.temperature = temperature
        self.sinkhorn_iters = sinkhorn_iters

    @property
    def codebook_size(self) -> int:
        return self.codebook.out_features

    @torch.no_grad()
    def extract_codebooks(self, x: torch.Tensor) -> torch.Tensor:
        x = self.head(x)
        x = F.normalize(x, p=2, dim=1)
        x = self.codebook(x)
        return F.softmax(x / self.temperature, dim=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Compute the loss. Input of shape (2*B, D), output of shape (B,).

        The batch is made like this:
        [x^1_orig, x^1_perturb, x^2_orig, x^2_perturb, ..., x^B_orig, x^B_perturb]
        """

        if x.size(0) % 2 != 0:
            raise ValueError("Batch size must be divisible by 2 to infer views.")
        if x.ndim != 2:
            raise ValueError("Input should be 2D.")

        x = self.head(x)
        x = F.normalize(x, p=2, dim=1)
        x = self.codebook(x)
        z1, z2 = x[::2], x[1::2]  # Split the batch by view
        log_p1 = F.log_softmax(z1 / self.temperature, dim=1)
        log_p2 = F.log_softmax(z2 / self.temperature, dim=1)
        with torch.no_grad():
            q1 = sinkhorn_knopp(z1, self.epsilon, self.sinkhorn_iters)
            q2 = sinkhorn_knopp(z2, self.epsilon, self.sinkhorn_iters)
        return -0.5 * ((q1 * log_p2).sum(dim=1) + (q2 * log_p1).sum(dim=1))
