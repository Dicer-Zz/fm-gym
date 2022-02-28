import math
import torch
from torch import nn, Tensor
from typing import Union


class MultiHeadAttention(nn.Module):
    """Construct the Multi Head Attention."""

    def __init__(
        self,
        num_attention_heads: int = 12,
        hidden_size: int = 768,
        attention_probs_dropout_prob: int = 0.1,
        position_embedding_type: str = "absolute",  # Supporting 'relative' is a bonus.
        is_cross_attention: bool = False,
    ) -> None:
        """
        Args:
            num_attention_heads (`int`, *optional*, defaults to 12):
                Number of attention heads for each attention layer in the Transformer encoder.
            hidden_size (`int`, *optional*, defaults to 768):
                Dimensionality of the encoder layers.
            attention_probs_dropout_prob (`float`, *optional*, defaults to 0.1):
                The dropout ratio for the attention probabilities.
            position_embedding_type (`str`, *optional*, defaults to `"absolute"`):
                Type of position embedding. Choose one of `"absolute"`, `"relative"`. For
                positional embeddings use `"absolute"`. For more information on `"relative"`, please refer to
                [Self-Attention with Relative Position Representations (Shaw et al.)](https://arxiv.org/abs/1803.02155).
        """
        super().__init__()

        assert hidden_size % num_attention_heads == 0
        assert (
            position_embedding_type == "absolute"
            or position_embedding_type == "relative"
        )

        self.num_attention_heads = num_attention_heads
        self.attention_head_size = hidden_size // num_attention_heads
        self.all_head_size = self.num_attention_heads * self.attention_head_size

        self.query = nn.Linear(hidden_size, self.all_head_size)
        self.key = nn.Linear(hidden_size, self.all_head_size)
        self.value = nn.Linear(hidden_size, self.all_head_size)

        self.dropout = nn.Dropout(attention_probs_dropout_prob)
        self.is_cross_attention = is_cross_attention

    def transpose_for_socres(self, x):
        # x.shape: (batch_size, seq_length, hidden_size)
        # new_x_shape: (batch_size, seq_length, num_attetion_heads, attention_head_size)
        # return
        new_x_shape = x.size()[:-1] + (
            self.num_attention_heads,
            self.attention_head_size,
        )
        x = x.view(new_x_shape)
        return x.permute(0, 2, 1, 3)
        

    def forward(
        self,
        hidden_states: Tensor,
        attention_mask: Tensor = None,
        encoder_hidden_states: Tensor = None,
        encoder_attention_mask: Tensor = None,
        output_attentions: bool = False,  # Return Tensor if not output_attentions else tuple(Tensor, Tensor)
    ):
        """
        Args:
            hidden_states (`torch.FloatTensor` of shape `(batch_size, sequence_length, hidden_size)`):
                Contextual representations.
            attention_mask (`torch.FloatTensor` of shape `(batch_size, sequence_length, sequence_length)`, *optional*):
                Mask to avoid performing attention on padding token & future token indices. Mask values selected in `[0, 1]`:
                - 1 for tokens that are **not masked**,
                - 0 for tokens that are **masked**.
            encoder_hidden_states (`torch.FloatTensor` of shape `(batch_size, sequence_length, hidden_size)`):
                Contextual representations of the encoding sequence.
            encoder_attention_mask (`torch.FloatTensor` of shape `(batch_size, sequence_length, sequence_length)`, *optional*):
                Mask to avoid performing attention on padding token indices. Mask values selected in `[0, 1]`:
                - 1 for tokens that are **not masked**,
                - 0 for tokens that are **masked**.
            output_attentions (`bool`, *optional*):
                Whether or not to return the attentions tensors of all attention layers.
        """
        self.multi_head_attention_type = 'cross-attention' if encoder_hidden_states else 'self-attention'
        pass
        self.multi_head_attention_type = (
            "cross-attention" if encoder_hidden_states else "self-attention"
        )
        if self.is_cross_attention:
            K = self.transpose_for_socres(self.key(encoder_hidden_states))
            V = self.transpose_for_socres(self.value(encoder_hidden_states))
            attention_mask = encoder_attention_mask
        else:
            K = self.transpose_for_socres(self.key(hidden_states))
            V = self.transpose_for_socres(self.value(hidden_states))

        Q = self.transpose_for_socres(self.query(hidden_states))

        attention_scores = torch.matmul(Q, K.transpose(-1, -2))
        # attention_scores.shape: (batch_size, num_attention_heads, seq_length, seq_length)

        attention_scores = attention_scores / math.sqrt(self.attention_head_size)

        # print(attention_scores.shape, attention_mask.shape)
        # # why?
        # if attention_mask is not None:
        #     attention_scores = attention_scores + attention_mask

        # Normalize
        attention_probs = nn.functional.softmax(attention_scores, dim=-1)
        attention_probs = self.dropout(attention_probs)

        context_layer = torch.matmul(attention_probs, V)

        # what is this?
        context_layer = context_layer.permute(0, 2, 1, 3).contiguous()
        new_context_layer_shape = context_layer.size()[:-2] + (self.all_head_size,)
        context_layer = context_layer.view(new_context_layer_shape)

        # outputs = (context_layer, attention_probs) if output_attentions else (context_layer,)

        return context_layer
    


class SelfAttentionSublayer(nn.Module):
    """
    Construct the Self-Attention Sublayer, consisting of an attention network,
    a layer normalization network, and a residual network.
    It is a part of the Transformer Encoder.
    """

    def __init__(
        self,
        num_attention_heads: int = 12,
        hidden_size: int = 768,
        attention_probs_dropout_prob: int = 0.1,
        layer_norm_eps: float = 1e-12,
        hidden_dropout_prob: float = 0.1,
        position_embedding_type: str = "absolute",  # Supporting 'relative' is a bonus.
    ) -> None:
        """
        Args:
            num_attention_heads (`int`, *optional*, defaults to 12):
                Number of attention heads for each attention layer in the Transformer encoder.
            hidden_size (`int`, *optional*, defaults to 768):
                Dimensionality of the encoder layers.
            attention_probs_dropout_prob (`float`, *optional*, defaults to 0.1):
                The dropout ratio for the attention probabilities.
            layer_norm_eps (`float`, *optional*, defaults to 1e-12):
                The epsilon used by the layer normalization layers.
            hidden_dropout_prob (`float`, *optional*, defaults to 0.1):
                The dropout probability for all fully connected layers in the embeddings and encoder.
            position_embedding_type (`str`, *optional*, defaults to `"absolute"`):
                Type of position embedding. Choose one of `"absolute"`, `"relative"`. For
                positional embeddings use `"absolute"`. For more information on `"relative"`, please refer to
                [Self-Attention with Relative Position Representations (Shaw et al.)](https://arxiv.org/abs/1803.02155).
        """
        super().__init__()
        assert (
            position_embedding_type == "absolute"
            or position_embedding_type == "relative"
        )
        self.atten = MultiHeadAttention(
            num_attention_heads,
            hidden_size,
            attention_probs_dropout_prob,
            position_embedding_type,
        )
        self.linear = nn.Linear(hidden_size, hidden_size)
        self.LayerNorm = nn.LayerNorm(hidden_size, layer_norm_eps)
        self.dropout = nn.Dropout(hidden_dropout_prob)
        

    def forward(
        self,
        hidden_states: Tensor,
        attention_mask: Tensor = None,
        output_attentions: bool = False,  # Return Tensor if not output_attentions else tuple(Tensor, Tensor)
    ):
        """
        Args:
            hidden_states (`torch.FloatTensor` of shape `(batch_size, sequence_length, hidden_size)`):
                Contextual representations.
            attention_mask (`torch.FloatTensor` of shape `(batch_size, sequence_length, sequence_length)`, *optional*):
                Mask to avoid performing attention on padding token indices. Mask values selected in `[0, 1]`:
                - 1 for tokens that are **not masked**,
                - 0 for tokens that are **masked**.
            output_attentions (`bool`, *optional*):
                Whether or not to return the attentions tensors of all attention layers.
        """
        pass
        outputs = self.atten(hidden_states, attention_mask, output_attentions)
        # print(f"selfAttetionOutputs: {outputs}")
        outputs = self.dropout(self.linear(outputs))
        outputs = self.LayerNorm(outputs + hidden_states)
        return outputs
    


class CrossAttentionSublayer(nn.Module):
    """
    Construct the Cross-Attention Sublayer, consisting of an attention network,
    a layer normalization network, and a residual network.
    It is a part of the Transformer Decoder.
    """

    def __init__(
        self,
        num_attention_heads: int = 12,
        hidden_size: int = 768,
        attention_probs_dropout_prob: int = 0.1,
        layer_norm_eps: float = 1e-12,
        hidden_dropout_prob: float = 0.1,
        position_embedding_type: str = "absolute",  # Supporting 'relative' is a bonus.
    ) -> None:
        """
        Args:
            num_attention_heads (`int`, *optional*, defaults to 12):
                Number of attention heads for each attention layer in the Transformer encoder.
            hidden_size (`int`, *optional*, defaults to 768):
                Dimensionality of the encoder layers.
            attention_probs_dropout_prob (`float`, *optional*, defaults to 0.1):
                The dropout ratio for the attention probabilities.
            layer_norm_eps (`float`, *optional*, defaults to 1e-12):
                The epsilon used by the layer normalization layers.
            hidden_dropout_prob (`float`, *optional*, defaults to 0.1):
                The dropout probability for all fully connected layers in the embeddings and encoder.
            position_embedding_type (`str`, *optional*, defaults to `"absolute"`):
                Type of position embedding. Choose one of `"absolute"`, `"relative"`. For
                positional embeddings use `"absolute"`. For more information on `"relative"`, please refer to
                [Self-Attention with Relative Position Representations (Shaw et al.)](https://arxiv.org/abs/1803.02155).
        """
        assert (
            position_embedding_type == "absolute"
            or position_embedding_type == "relative"
        )
        pass

    def forward(
        self,
        hidden_states: Tensor,
        attention_mask: Tensor = None,
        encoder_hidden_states: Tensor = None,
        encoder_attention_mask: Tensor = None,
        output_attentions: bool = False,  # Return Tensor if not output_attentions else tuple(Tensor, Tensor)
    ):
        """
        Args:
            hidden_states (`torch.FloatTensor` of shape `(batch_size, sequence_length, hidden_size)`):
                Contextual representations.
            attention_mask (`torch.FloatTensor` of shape `(batch_size, sequence_length, sequence_length)`, *optional*):
                Mask to avoid performing attention on padding token & future token indices. Mask values selected in `[0, 1]`:
                - 1 for tokens that are **not masked**,
                - 0 for tokens that are **masked**.
            encoder_hidden_states (`torch.FloatTensor` of shape `(batch_size, sequence_length, hidden_size)`):
                Contextual representations of the encoding sequence.
            encoder_attention_mask (`torch.FloatTensor` of shape `(batch_size, sequence_length, sequence_length)`, *optional*):
                Mask to avoid performing attention on padding token indices. Mask values selected in `[0, 1]`:
                - 1 for tokens that are **not masked**,
                - 0 for tokens that are **masked**.
            output_attentions (`bool`, *optional*):
                Whether or not to return the attentions tensors of all attention layers.
        """
        pass


class BertSelfAttention(nn.Module):
    def __init__(
        self,
        num_attention_heads: int = 12,
        hidden_size: int = 768,
        attention_probs_dropout_prob: int = 0.1,
        position_embedding_type: str = "absolute",  # Supporting 'relative' is a bonus.
    ):
        super().__init__()
        assert hidden_size % num_attention_heads == 0

        self.num_attention_heads = num_attention_heads
        self.attention_head_size = int(hidden_size / num_attention_heads)
        self.all_head_size = self.num_attention_heads * self.attention_head_size

        self.query = nn.Linear(hidden_size, self.all_head_size)
        self.key = nn.Linear(hidden_size, self.all_head_size)
        self.value = nn.Linear(hidden_size, self.all_head_size)

        self.dropout = nn.Dropout(attention_probs_dropout_prob)

    def transpose_for_scores(self, x):
        new_x_shape = x.size()[:-1] + (
            self.num_attention_heads,
            self.attention_head_size,
        )
        x = x.view(new_x_shape)
        return x.permute(0, 2, 1, 3)

    def forward(
        self,
        hidden_states,
        attention_mask=None,
        head_mask=None,
        encoder_hidden_states=None,
        encoder_attention_mask=None,
        past_key_value=None,
        output_attentions=False,
    ):
        mixed_query_layer = self.query(hidden_states)

        # If this is instantiated as a cross-attention module, the keys
        # and values come from an encoder; the attention mask needs to be
        # such that the encoder's padding tokens are not attended to.
        is_cross_attention = encoder_hidden_states is not None

        if is_cross_attention and past_key_value is not None:
            # reuse k,v, cross_attentions
            key_layer = past_key_value[0]
            value_layer = past_key_value[1]
            attention_mask = encoder_attention_mask
        elif is_cross_attention:
            key_layer = self.transpose_for_scores(self.key(encoder_hidden_states))
            value_layer = self.transpose_for_scores(self.value(encoder_hidden_states))
            attention_mask = encoder_attention_mask
        elif past_key_value is not None:
            key_layer = self.transpose_for_scores(self.key(hidden_states))
            value_layer = self.transpose_for_scores(self.value(hidden_states))
            key_layer = torch.cat([past_key_value[0], key_layer], dim=2)
            value_layer = torch.cat([past_key_value[1], value_layer], dim=2)
        else:
            key_layer = self.transpose_for_scores(self.key(hidden_states))
            value_layer = self.transpose_for_scores(self.value(hidden_states))

        query_layer = self.transpose_for_scores(mixed_query_layer)

        # Take the dot product between "query" and "key" to get the raw attention scores.
        attention_scores = torch.matmul(query_layer, key_layer.transpose(-1, -2))

        attention_scores = attention_scores / math.sqrt(self.attention_head_size)
        if attention_mask is not None:
            # Apply the attention mask is (precomputed for all layers in BertModel forward() function)
            attention_scores = attention_scores + attention_mask

        # Normalize the attention scores to probabilities.
        attention_probs = nn.functional.softmax(attention_scores, dim=-1)

        # This is actually dropping out entire tokens to attend to, which might
        # seem a bit unusual, but is taken from the original Transformer paper.
        attention_probs = self.dropout(attention_probs)

        context_layer = torch.matmul(attention_probs, value_layer)

        context_layer = context_layer.permute(0, 2, 1, 3).contiguous()
        new_context_layer_shape = context_layer.size()[:-2] + (self.all_head_size,)
        context_layer = context_layer.view(new_context_layer_shape)

        return context_layer


class BertSelfOutput(nn.Module):
    def __init__(
        self,
        hidden_size: int = 768,
        layer_norm_eps: float = 1e-12,
        hidden_dropout_prob: float = 0.1,
    ):
        super().__init__()
        self.dense = nn.Linear(hidden_size, hidden_size)
        self.LayerNorm = nn.LayerNorm(hidden_size, eps=layer_norm_eps)
        self.dropout = nn.Dropout(hidden_dropout_prob)

    def forward(self, hidden_states, input_tensor):
        hidden_states = self.dense(hidden_states)
        hidden_states = self.dropout(hidden_states)
        hidden_states = self.LayerNorm(hidden_states + input_tensor)
        return hidden_states


class BertAttention(nn.Module):
    def __init__(
        self,
        num_attention_heads: int = 12,
        hidden_size: int = 768,
        attention_probs_dropout_prob: int = 0.1,
        position_embedding_type: str = "absolute",  # Supporting 'relative' is a bonus.
        layer_norm_eps: float = 1e-12,
        hidden_dropout_prob: float = 0.1,
    ):
        super().__init__()
        self.self = BertSelfAttention(
            num_attention_heads,
            hidden_size,
            attention_probs_dropout_prob,
            position_embedding_type=position_embedding_type,
        )
        self.output = BertSelfOutput(hidden_size, layer_norm_eps, hidden_dropout_prob)

    def forward(
        self,
        hidden_states,
        attention_mask=None,
        head_mask=None,
        encoder_hidden_states=None,
        encoder_attention_mask=None,
        past_key_value=None,
        output_attentions=False,
    ):
        self_outputs = self.self(
            hidden_states,
            attention_mask,
            head_mask,
            encoder_hidden_states,
            encoder_attention_mask,
            past_key_value,
            output_attentions,
        )
        # print(self_outputs)
        attention_output = self.output(self_outputs, hidden_states)
        return attention_output


if __name__ == "__main__":
    torch.manual_seed(42)
    torch.cuda.manual_seed(42)

    atten = MultiHeadAttention()
    input_tensors = torch.rand((10, 20, 768))
    atten1 = atten(input_tensors)

    torch.manual_seed(42)
    torch.cuda.manual_seed(42)

    atten = BertSelfAttention()
    input_tensors = torch.rand((10, 20, 768))
    atten2 = atten(input_tensors)
    # assert atten1 == atten2

    # MultiHeadAttention testing pass!

    torch.manual_seed(42)
    torch.cuda.manual_seed(42)

    selfAtten = SelfAttentionSublayer()
    input_tensors = torch.rand((10, 20, 768))
    atten1 = selfAtten(input_tensors)
    print(atten1.shape, atten1)

    torch.manual_seed(42)
    torch.cuda.manual_seed(42)

    selfAtten = BertAttention()
    input_tensors = torch.rand((10, 20, 768))
    atten2 = selfAtten(input_tensors)
    print(atten2.shape, atten2)

    # SelfAttentionSublayer testing pass!