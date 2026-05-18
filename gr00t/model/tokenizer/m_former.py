from transformers.models.vit.modeling_vit import (
    ViTConfig,
    ViTPreTrainedModel,
    ViTEncoder
)
from torch import nn
import os
import torch
from typing import Optional, Dict, List, Tuple, Union
from transformers.modeling_outputs import BaseModelOutput


class MFormerEmbeddings(nn.Module):
    def __init__(self, config: ViTConfig) -> None:
        super().__init__()
        query_num = config.query_num
        self.query_num = query_num
        self.latent_motion_token = nn.Parameter(torch.zeros(1, query_num, config.hidden_size))
        self.sep_token = nn.Parameter(torch.zeros(1, 1, config.hidden_size))
        self.projection = nn.Linear(config.input_hidden_size, config.hidden_size, bias=True)
        
        # Support manual max_seq_len specification (backward compatible)
        if hasattr(config, "max_seq_len") and config.max_seq_len is not None:
            # Use manually specified max sequence length
            max_seq_len = config.max_seq_len
        else:
            # Default: num_patches*2 + 1 + query_num (for vision tasks)
            max_seq_len = config.num_patches*2 + 1 + query_num
        
        self.max_seq_len = max_seq_len
        self.position_embeddings = nn.Parameter(torch.randn(1, max_seq_len, config.hidden_size))
        self.token_type_embeddings = nn.Parameter(torch.randn(2, config.hidden_size))
        self.dropout = nn.Dropout(config.hidden_dropout_prob)
        self.config = config
        if hasattr(config, "legacy"):
            self.legacy = config.legacy
        else:
            self.legacy = True

    def forward(
        self,
        cond_hidden_states: torch.Tensor,
        target_hidden_states: torch.Tensor,
    ) -> torch.Tensor:
        # Debug/validate shapes: expect [B, T, D]
        if cond_hidden_states.dim() != 3 or target_hidden_states.dim() != 3:
            raise RuntimeError(
                f"MFormerEmbeddings expects 3D tensors [B,T,D]. Got cond_hidden_states={tuple(cond_hidden_states.shape)}, "
                f"target_hidden_states={tuple(target_hidden_states.shape)}. "
                f"Upstream encoder likely returned [B,D] (pooled/squeezed). Ensure token-level features."
            )
        batch_size, per_seq_length = cond_hidden_states.shape[:2]

        cond_embeddings = self.projection(cond_hidden_states)

        latent_motion_tokens = self.latent_motion_token.expand(batch_size, -1, -1)
        sep_tokens = self.sep_token.expand(batch_size, -1, -1)
        # Optional runtime diagnostics
        if os.environ.get("LMT_DEBUG_SHAPES", "0") == "1":
            print(
                {
                    "latent_motion_tokens": tuple(latent_motion_tokens.shape),
                    "cond_embeddings_before_cat": tuple(cond_embeddings.shape),
                    "sep_tokens": tuple(sep_tokens.shape),
                }
            )
        cond_embeddings = torch.cat((latent_motion_tokens, cond_embeddings, sep_tokens), dim=1)

        target_embeddings = self.projection(target_hidden_states)
        embeddings = torch.cat((cond_embeddings, target_embeddings), dim=1)

        # add positional encoding to each token
        embeddings = embeddings + self.position_embeddings[:, :embeddings.shape[1], :]

        # add token type encoding to each token
        cond_token_type_embeddings = self.token_type_embeddings[0].expand(batch_size, per_seq_length + self.query_num + 1, -1)
        if self.legacy:
            target_token_type_embeddings = self.token_type_embeddings[0].expand(batch_size, target_embeddings.shape[1], -1)
        else:
            target_token_type_embeddings = self.token_type_embeddings[1].expand(batch_size, target_embeddings.shape[1], -1)
        token_type_embeddings = torch.cat((cond_token_type_embeddings, target_token_type_embeddings), dim=1)
        embeddings = embeddings + token_type_embeddings

        embeddings = self.dropout(embeddings)

        return embeddings

class MFormer(ViTPreTrainedModel):
    def __init__(self, config: ViTConfig):
        super().__init__(config)
        self.config = config
        self.query_num = config.query_num
        self.embeddings = MFormerEmbeddings(config)
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
        elif isinstance(module, MFormerEmbeddings):
            module.position_embeddings.data = nn.init.trunc_normal_(
                module.position_embeddings.data.to(torch.float32),
                mean=0.0,
                std=self.config.initializer_range,
            ).to(module.position_embeddings.dtype)

            module.token_type_embeddings.data = nn.init.trunc_normal_(
                module.token_type_embeddings.data.to(torch.float32),
                mean=0.0,
                std=self.config.initializer_range,
            ).to(module.token_type_embeddings.dtype)

            module.latent_motion_token.data = nn.init.trunc_normal_(
                module.latent_motion_token.data.to(torch.float32),
                mean=0.0,
                std=self.config.initializer_range,
            ).to(module.latent_motion_token.dtype)

            module.sep_token.data = nn.init.trunc_normal_(
                module.sep_token.data.to(torch.float32),
                mean=0.0,
                std=self.config.initializer_range,
            ).to(module.sep_token.dtype)

    # def get_input_embeddings(self):
    #     return self.embeddings.projection

    def _prune_heads(self, heads_to_prune: Dict[int, List[int]]) -> None:
        """
        Prunes heads of the model. heads_to_prune: dict of {layer_num: list of heads to prune in this layer} See base
        class PreTrainedModel
        """
        for layer, heads in heads_to_prune.items():
            self.encoder.layer[layer].attention.prune_heads(heads)


    def forward(
        self,
        cond_hidden_states: torch.Tensor,
        target_hidden_states: torch.Tensor,
        head_mask: Optional[torch.Tensor] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
    ) -> Union[Tuple, BaseModelOutput]:
        r"""
        bool_masked_pos (`torch.BoolTensor` of shape `(batch_size, num_patches)`, *optional*):
            Boolean masked positions. Indicates which patches are masked (1) and which aren't (0).
        """
        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        # Prepare head mask if needed
        # 1.0 in head_mask indicate we keep the head
        # attention_probs has shape bsz x n_heads x N x N
        # input head_mask has shape [num_heads] or [num_hidden_layers x num_heads]
        # and head_mask is converted to shape [num_hidden_layers x batch x num_heads x seq_length x seq_length]
        head_mask = self.get_head_mask(head_mask, self.config.num_hidden_layers)


        embedding_output = self.embeddings(
            cond_hidden_states=cond_hidden_states,
            target_hidden_states=target_hidden_states
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


