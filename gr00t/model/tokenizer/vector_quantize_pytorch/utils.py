import torch
from torch import nn
from torch.nn import Module, ModuleList

# quantization

from .vector_quantize_pytorch import VectorQuantize
from .residual_vq import ResidualVQ, GroupedResidualVQ
from .random_projection_quantizer import RandomProjectionQuantizer
from .finite_scalar_quantization import FSQ
from .lookup_free_quantization import LFQ
from .residual_lfq import ResidualLFQ, GroupedResidualLFQ
from .residual_fsq import ResidualFSQ, GroupedResidualFSQ
from .latent_quantization import LatentQuantize
from .sim_vq import SimVQ
from .residual_sim_vq import ResidualSimVQ

QUANTIZE_KLASSES = (
    VectorQuantize,
    ResidualVQ,
    GroupedResidualVQ,
    RandomProjectionQuantizer,
    FSQ,
    LFQ,
    SimVQ,
    ResidualSimVQ,
    ResidualLFQ,
    GroupedResidualLFQ,
    ResidualFSQ,
    GroupedResidualFSQ,
    LatentQuantize
)

# classes

class Sequential(Module):
    def __init__(
        self,
        *fns: Module
    ):
        super().__init__()
        assert sum([int(isinstance(fn, QUANTIZE_KLASSES)) for fn in fns]) == 1, 'this special Sequential must contain exactly one quantizer'

        self.fns = ModuleList(fns)

    def forward(
        self,
        x,
        **kwargs
    ):
        for fn in self.fns:

            if not isinstance(fn, QUANTIZE_KLASSES):
                x = fn(x)
                continue

            x, *rest = fn(x, **kwargs)

        output = (x, *rest)

        return output
