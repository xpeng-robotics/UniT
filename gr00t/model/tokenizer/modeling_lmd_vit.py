from transformers.models.vit.modeling_vit import (
    ViTPatchEmbeddings,
    ViTConfig,
    ViTPreTrainedModel,
    ViTEncoder
)
from torch import nn
import torch
from typing import Optional, Dict, List, Tuple, Union
from transformers.modeling_outputs import BaseModelOutput


class LMDViTEmbeddings(nn.Module):
    def __init__(self, config: ViTConfig) -> None:
        super().__init__()
        query_num = config.query_num
        self.query_num = query_num

        is_io_hidden_states = getattr(config, 'is_io_hidden_states', False)
        if is_io_hidden_states:
            self.patch_embeddings = nn.Linear(config.hidden_size, config.hidden_size, bias=True)
        else:
            self.patch_embeddings = ViTPatchEmbeddings(config)

        query_fusion_mode = getattr(config, 'query_fusion_mode', 'add')
        assert query_fusion_mode in ['add', 'concat']
        self.query_fusion_mode = query_fusion_mode
        self.query_pooling_layer = nn.Linear(query_num * config.hidden_size, config.hidden_size) if query_fusion_mode == 'add' else None

        seq_len = config.num_patches + (query_fusion_mode == 'concat') * query_num
        self.position_embeddings = nn.Parameter(torch.randn(1, seq_len, config.hidden_size))
        self.dropout = nn.Dropout(config.hidden_dropout_prob)
        self.config = config

    def forward(
        self,
        pixel_values: torch.Tensor,
        latent_motion_tokens: torch.Tensor,
    ) -> torch.Tensor:
        embeddings = self.patch_embeddings(pixel_values)

        if self.query_fusion_mode == 'add':
            # add the projected motion token to the embedded patch tokens
            assert self.query_pooling_layer is not None  # gated by fusion mode above
            latent_motion_tokens = latent_motion_tokens.reshape(
                latent_motion_tokens.shape[0], 1, -1
            )
            latent_motion_tokens = self.query_pooling_layer(latent_motion_tokens)
            embeddings = embeddings + latent_motion_tokens
        elif self.query_fusion_mode == 'concat':
            # concatenate latent motion tokens with the embedded patch tokens
            embeddings = torch.cat((latent_motion_tokens, embeddings), dim=1)

        # add positional encoding to each token
        embeddings = embeddings + self.position_embeddings

        embeddings = self.dropout(embeddings)

        return embeddings


class LMDViTModel(ViTPreTrainedModel):
    def __init__(self, config: ViTConfig):
        super().__init__(config)
        self.config = config
        is_io_hidden_states = getattr(config, 'is_io_hidden_states', False)
        self.is_io_hidden_states = is_io_hidden_states

        self.embeddings = LMDViTEmbeddings(config)
        self.encoder = ViTEncoder(config)

        self.layernorm = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)

        self.post_init()

    def _init_weights(self, module: Union[nn.Linear, nn.Conv2d, nn.LayerNorm]) -> None:
        """Initialize the weights"""
        if isinstance(module, (nn.Linear, nn.Conv2d)):
            # Upcast the input in `fp32` and cast it back to desired `dtype` to avoid
            # `trunc_normal_cpu` not implemented in `half` issues
            module.weight.data = nn.init.trunc_normal_(
                module.weight.data.to(torch.float32), mean=0.0, std=self.config.initializer_range
            ).to(module.weight.dtype)
            if module.bias is not None:
                module.bias.data.zero_()
        elif isinstance(module, nn.LayerNorm):
            module.bias.data.zero_()
            module.weight.data.fill_(1.0)
        elif isinstance(module, LMDViTEmbeddings):
            module.position_embeddings.data = nn.init.trunc_normal_(
                module.position_embeddings.data.to(torch.float32),
                mean=0.0,
                std=self.config.initializer_range,
            ).to(module.position_embeddings.dtype)

    def get_input_embeddings(self) -> Union[ViTPatchEmbeddings, nn.Linear]:
        """Patch projector: ViT convolution (pixels) or Linear (pre-encoded patch features)."""
        return self.embeddings.patch_embeddings

    def _prune_heads(self, heads_to_prune: Dict[int, List[int]]) -> None:
        """
        Prunes heads of the model. heads_to_prune: dict of {layer_num: list of heads to prune in this layer} See base
        class PreTrainedModel
        """
        for layer, heads in heads_to_prune.items():
            self.encoder.layer[layer].attention.prune_heads(heads)


    def forward(
        self,
        pixel_values: Optional[torch.Tensor] = None,
        latent_motion_tokens: Optional[torch.Tensor] = None, # change lam_tokens to latent_motion_tokens
        head_mask: Optional[torch.Tensor] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
    ) -> Union[Tuple, BaseModelOutput]:
        r"""Encode fused condition (patch features + latent motion tokens) with ViTEncoder."""
        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        if pixel_values is None:
            raise ValueError("You have to specify pixel_values")

        # Prepare head mask if needed
        # 1.0 in head_mask indicate we keep the head
        # attention_probs has shape bsz x n_heads x N x N
        # input head_mask has shape [num_heads] or [num_hidden_layers x num_heads]
        # and head_mask is converted to shape [num_hidden_layers x batch x num_heads x seq_length x seq_length]
        head_mask = self.get_head_mask(head_mask, self.config.num_hidden_layers)

        # TODO: maybe have a cleaner way to cast the input (from `ImageProcessor` side?)
        if self.is_io_hidden_states:
            expected_dtype = self.embeddings.patch_embeddings.weight.dtype
        else:
            expected_dtype = self.embeddings.patch_embeddings.projection.weight.dtype
        if pixel_values.dtype != expected_dtype:
            pixel_values = pixel_values.to(expected_dtype)

        embedding_output = self.embeddings(
            pixel_values=pixel_values,
            latent_motion_tokens=latent_motion_tokens
        )

        encoder_outputs = self.encoder(
            embedding_output,
            head_mask=head_mask,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )
        sequence_output = encoder_outputs[0]
        sequence_output = self.layernorm(sequence_output)

        if not return_dict:
            return (sequence_output,) + encoder_outputs[1:]

        return BaseModelOutput(
            last_hidden_state=sequence_output,
            hidden_states=encoder_outputs.hidden_states,
            attentions=encoder_outputs.attentions,
        )
