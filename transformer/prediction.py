from torch import Tensor
from torch import nn

from utils import act2fn

class BERTMLMHead(nn.Module):
    """
    Construct the BERT Masked Language Model Prediction Head:
    Linear -> LayerNorm -> Linear
    """
    def __init__(
        self,
        hidden_size: int = 768,
        mlm_head_act: str = "gelu",
        vocab_size: int = 30522,
        layer_norm_eps: float = 1e-12,
    ) -> None:
        super().__init__()
        self.transform = nn.Linear(hidden_size, hidden_size)
        self.decoder = nn.Linear(hidden_size, vocab_size)
        self.act_fn = act2fn[mlm_head_act]()
        self.LayerNorm = nn.LayerNorm(hidden_size, layer_norm_eps)

    def forward(
        self,
        hidden_states: Tensor,
    ) -> Tensor:
        outputs = self.transform(hidden_states)
        outputs = self.act_fn(outputs)
        outputs = self.LayerNorm(outputs)
        outputs = self.decoder(outputs)
        return outputs


class BERTNSPHead(nn.Module):
    """
    Construct the BERT Next Sentence Prediction Head:
    Linear -> Linear
    """

    def __init__(
        self,
        hidden_size: int = 768,
        nsp_head_act: str = "tanh",
        label_size: int = 2,
    ) -> None:
        super().__init__()
        self.linear = nn.Linear(hidden_size, label_size)
        self.act_fn = act2fn[nsp_head_act]()

    def forward(
        self,
        hidden_states: Tensor,
    ) -> Tensor:
        outputs = self.linear(hidden_states)
        outputs = self.act_fn(outputs)
        return outputs


