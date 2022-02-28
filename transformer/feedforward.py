import torch
from torch import Tensor
from torch import nn


class FeedforwardSublayer(nn.Module):
    """
    Construct the Feedforward Sublayer, consisting of two linear networks,
    a layer normalization network, and a residual network.
    """

    def __init__(
        self,
        hidden_size: int = 768,
        intermediate_size: int = 3072,
        hidden_act: str = "gelu",
        layer_norm_eps: float = 1e-12,
        hidden_dropout_prob: float = 0.1,
    ) -> None:
        """
        Args:
            hidden_size (`int`, *optional*, defaults to 768):
                Dimensionality of the encoder layers.
            intermediate_size (`int`, *optional*, defaults to 3072):
                Dimensionality of the "intermediate" (often named feed-forward) layer in the Transformer encoder.
            hidden_act (`str` or `Callable`, *optional*, defaults to `"gelu"`):
                The non-linear activation function (function or string) in the encoder.
            layer_norm_eps (`float`, *optional*, defaults to 1e-12):
                The epsilon used by the layer normalization layers.
            hidden_dropout_prob (`float`, *optional*, defaults to 0.1):
                The dropout probability for all fully connected layers in the embeddings and encoder.
        """
        super().__init__()
        self.act2fn = {
            "relu": nn.ReLU,
            "gelu": nn.GELU,
        }
        self.linear1 = nn.Linear(hidden_size, intermediate_size)
        self.linear2 = nn.Linear(intermediate_size, hidden_size)

        self.hidden_act = self.act2fn[hidden_act]()
        self.LayerNorm = nn.LayerNorm(hidden_size, layer_norm_eps)
        self.dropout = nn.Dropout(hidden_dropout_prob)

    def forward(self, hidden_states: Tensor) -> Tensor:
        """
        Args:
            hidden_states (`torch.FloatTensor` of shape `(batch_size, sequence_length, hidden_size)`):
                Contextual representations.
        """
        # ! processing order
        residual = hidden_states

        hidden_states = self.linear2(self.hidden_act(self.linear1(hidden_states)))
        hidden_states = self.dropout(hidden_states)

        hidden_states += residual
        hidden_states = self.LayerNorm(hidden_states)
        return hidden_states


class BertIntermediate(nn.Module):
    def __init__(
        self, 
        hidden_size=768, 
        intermediate_size=3072, 
        hidden_act="gelu"
    ):
        super().__init__()
        self.act2fn = {
            "relu": nn.ReLU,
            "gelu": nn.GELU,
        }
        self.dense = nn.Linear(hidden_size, intermediate_size)
        if isinstance(hidden_act, str):
            self.intermediate_act_fn = self.act2fn[hidden_act]()
        else:
            self.intermediate_act_fn = hidden_act

    def forward(self, hidden_states):
        hidden_states = self.dense(hidden_states)
        hidden_states = self.intermediate_act_fn(hidden_states)
        return hidden_states


class BertOutput(nn.Module):
    def __init__(
        self, 
        hidden_size=768, 
        intermediate_size=3072, 
        layer_norm_eps=1e-12,
        hidden_dropout_prob=0.1,
    ):
        super().__init__()
        self.dense = nn.Linear(intermediate_size, hidden_size)
        self.LayerNorm = nn.LayerNorm(hidden_size, eps=layer_norm_eps)
        self.dropout = nn.Dropout(hidden_dropout_prob)

    def forward(self, hidden_states, input_tensor):
        hidden_states = self.dense(hidden_states)
        hidden_states = self.dropout(hidden_states)
        hidden_states = self.LayerNorm(hidden_states + input_tensor)
        return hidden_states


if __name__ == "__main__":
    torch.manual_seed(42)
    torch.cuda.manual_seed(42)

    x = torch.rand((512, 768))
    print(x)
    ffn = FeedforwardSublayer()
    print(ffn(x))

    torch.manual_seed(42)
    torch.cuda.manual_seed(42)
    x = torch.rand((512, 768))
    print(x)
    ffn1 = BertIntermediate()
    ffn2 = BertOutput()
    print(ffn2(ffn1(x), x))
    
    # pass testing
