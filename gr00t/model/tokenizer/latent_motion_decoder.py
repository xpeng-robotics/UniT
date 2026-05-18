from transformers.models.vit.modeling_vit import (
    ViTConfig,
)
from torch import nn
import torch
import math
from typing import Optional
from .modeling_lmd_vit import LMDViTModel


class LatentMotionDecoder(nn.Module):
    def __init__(self,
                 config: ViTConfig) -> None:
        super().__init__()
        self.config = config
        self.transformer = LMDViTModel(config)

        is_io_hidden_states = getattr(config, 'is_io_hidden_states', False)
        self.is_io_hidden_states = is_io_hidden_states

        if is_io_hidden_states:
            self.decoder = nn.Linear(config.hidden_size, config.hidden_size, bias=True)

            self.decoder.weight.data = nn.init.trunc_normal_(
                self.decoder.weight.data.to(torch.float32), mean=0.0, std=self.config.initializer_range
            ).to(self.decoder.weight.dtype)
            self.decoder.bias.data.zero_()

        else:
            self.decoder = nn.Sequential(
                nn.Conv2d(
                    in_channels=config.hidden_size,
                    out_channels=config.encoder_stride**2 * config.num_channels,
                    kernel_size=1,
                ),
                nn.PixelShuffle(config.encoder_stride),
            )

            self.decoder[0].weight.data = nn.init.trunc_normal_(
                self.decoder[0].weight.data.to(torch.float32), mean=0.0, std=self.config.initializer_range
            ).to(self.decoder[0].weight.dtype)


    def forward(
        self,
        cond_input: Optional[torch.Tensor] = None,
        latent_motion_tokens: Optional[torch.Tensor] = None
    ) -> torch.Tensor:

        outputs = self.transformer(
            pixel_values=cond_input,
            latent_motion_tokens=latent_motion_tokens
        )

        sequence_output = outputs[0]
        sequence_output = sequence_output[:, -self.config.num_patches:]
        batch_size, sequence_length, num_channels = sequence_output.shape

        if not self.is_io_hidden_states:
            height = width = math.floor(sequence_length**0.5)
            sequence_output = sequence_output.permute(0, 2, 1).contiguous().reshape(batch_size, num_channels, height, width)

        reconstructed_output = self.decoder(sequence_output)

        return reconstructed_output
