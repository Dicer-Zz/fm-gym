import torch
from typing import Union
from torch import Tensor
from torch import nn

from attention import SelfAttentionSublayer
from feedforward import FeedforwardSublayer


class XformerLayer(nn.Module):
    """
    Construct the Layer of Transformer Encoder,
    cosisting of a self-attention network and a feedforward network.
    """

    def __init__(
        self,
        num_attention_heads: int = 12,
        hidden_size: int = 768,
        attention_probs_dropout_prob: int = 0.1,
        layer_norm_eps: float = 1e-12,
        hidden_dropout_prob: float = 0.1,
        position_embedding_type: str = "absolute",  # Supporting 'relative' is a bonus.
        intermediate_size: int = 3072,
        hidden_act: str = "gelu",
    ):
        super().__init__()
        self.selfAtten = SelfAttentionSublayer(
            num_attention_heads,
            hidden_size,
            attention_probs_dropout_prob,
            layer_norm_eps,
            hidden_dropout_prob,
            position_embedding_type,
        )
        self.ffn = FeedforwardSublayer(
            hidden_size, intermediate_size, hidden_act, layer_norm_eps
        )

    def forward(
        self,
        hidden_states: Tensor,
        attention_mask: Tensor = None,
        output_attentions: bool = False,
        output_hidden_states: bool = False,
    ):
        outputs = self.selfAtten(hidden_states, attention_mask, output_attentions)
        outputs = self.ffn(outputs)
        return outputs


class XformerEncoder(nn.Module):
    """
    Construct the Transformer Encoder,
    consisting of a list of a tuple of a self-attention network and a feedforward network.
    """

    def __init__(
        self,
        num_hidden_layers: int = 12,
        num_attention_heads: int = 12,
        hidden_size: int = 768,
        attention_probs_dropout_prob: int = 0.1,
        layer_norm_eps: float = 1e-12,
        hidden_dropout_prob: float = 0.1,
        position_embedding_type: str = "absolute",  # Supporting 'relative' is a bonus.
        intermediate_size: int = 3072,
        hidden_act: str = "gelu",
    ) -> None:
        super().__init__()
        self.layers = nn.ModuleList(
            [
                XformerLayer(
                    num_attention_heads,
                    hidden_size,
                    attention_probs_dropout_prob,
                    layer_norm_eps,
                    hidden_dropout_prob,
                    position_embedding_type,  # Supporting 'relative' is a bonus.
                    intermediate_size,
                    hidden_act,
                )
            for _ in range(num_hidden_layers)]
        )
        

    def forward(
        self,
        hidden_states: Tensor,
        attention_mask: Tensor = None,
        output_attentions: bool = False,
        output_hidden_states: bool = False,
    ):
        for _, layer_module in enumerate(self.layers):
            layer_outputs = layer_module(
                hidden_states, 
                attention_mask, 
                output_attentions, 
                output_hidden_states
            )

            hidden_states = layer_outputs

        return hidden_states

if __name__ == "__main__":
    torch.manual_seed(42)
    torch.cuda.manual_seed(42)

    bert = XformerEncoder()
    input_tensors = torch.rand((10, 20, 768))

    outputs = bert(input_tensors)
    print(outputs.shape, outputs)