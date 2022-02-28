import torch
from torch import nn
from torch import Tensor, LongTensor


class LearnableEmbeddings(nn.Module):
    """Construct the Learnable embeddings."""
    def __init__(
        self,
        num_embeddings: int,
        embedding_dim: int,
        pad_token_id: int,
    ) -> None:
        super().__init__()
        self.word_embeddings = nn.Embedding(num_embeddings, embedding_dim, padding_idx=pad_token_id)

    def forward(
        self,
        ids: Tensor,
    ) -> Tensor:
        word_embeddings = self.word_embeddings(ids)
        return word_embeddings

class SinusoidalEmbeddings(nn.Module):
    """Construct the Sinusoidal embeddings."""
    def __init__(
        self,
        num_embeddings: int,
        embedding_dim: int,
        pad_token_id: int,
        div_scale: int = 10000,
    ) -> None:
        self.num_embeddings = num_embeddings
        self.embedding_dim = embedding_dim
        self.pad_token_id = pad_token_id
        self.div_scale = div_scale
        self.postion_embeddings = nn.Embedding(num_embeddings, embedding_dim, pad_token_id)        

    def forward(
        self,
        ids: Tensor,
    ) -> Tensor:
        pass


class BERTEmbeddings(nn.Module):
    """Construct the embeddings from word, position and token_type embeddings."""
    def __init__(
        self,
        vocab_size: int = 30522,
        hidden_size: int = 768,
        pad_token_id: int = 0,
        max_position_embeddings: int = 512,
        type_vocab_size: int = 2,
        layer_norm_eps: float = 1e-12,
        hidden_dropout_prob: float = 0.1,
    ) -> None:
        """
        Args:
            vocab_size (`int`, *optional*, defaults to 30522):
                Vocabulary size of the BERT model. Defines the number of different tokens that can be represented by the
                `inputs_ids`.
            hidden_size (`int`, *optional*, defaults to 768):
                Dimensionality of the encoder layers.
            max_position_embeddings (`int`, *optional*, defaults to 512):
                The maximum sequence length that this model might ever be used with. Typically set this to something large
                just in case (e.g., 512 or 1024 or 2048).
            type_vocab_size (`int`, *optional*, defaults to 2):
                The vocabulary size of the `token_type_ids`.
            layer_norm_eps (`float`, *optional*, defaults to 1e-12):
                The epsilon used by the layer normalization layers.
            hidden_dropout_prob (`float`, *optional*, defaults to 0.1):
                The dropout probability for all fully connected layers in the embeddings and encoder.
        """
        super().__init__()
        self.word_embeddings = nn.Embedding(vocab_size, hidden_size, padding_idx=pad_token_id)
        self.position_embeddings = nn.Embedding(max_position_embeddings, hidden_size)
        self.token_type_embeddings = nn.Embedding(type_vocab_size, hidden_size)

        self.LayerNorm = nn.LayerNorm(hidden_size, layer_norm_eps)
        self.dropout = nn.Dropout(hidden_dropout_prob)

        self.register_buffer("position_ids", torch.arange(max_position_embeddings).expand(1, -1))
        self.register_buffer("token_type_ids", torch.zeros(self.position_ids.size(), dtype=torch.long), persistent=False)

    def forward(
        self,
        input_ids: Tensor,
        token_type_ids: Tensor = None,
        position_ids: Tensor = None,
    ) -> Tensor:
        """
        Args:
            input_ids (`torch.LongTensor` of shape `(batch_size, sequence_length)`):
                Indices of input sequence tokens in the vocabulary.
                Indices can be obtained using [`BertTokenizer`].
            token_type_ids (`torch.LongTensor` of shape `(batch_size, sequence_length)`, *optional*):
                Segment token indices to indicate first and second portions of the inputs. Indices are selected in `[0,
                1]`:
                - 0 corresponds to a *sentence A* token,
                - 1 corresponds to a *sentence B* token.
            position_ids (`torch.LongTensor` of shape `(batch_size, sequence_length)`, *optional*):
                Indices of positions of each input sequence tokens in the position embeddings. Selected in the range `[0,
                config.max_position_embeddings - 1]`.
        """
        input_shape = input_ids.size()
        seq_length = input_shape[1]

        if position_ids is None:
            position_ids = self.position_ids[:, :seq_length]
        
        if token_type_ids is None:
            buffered_token_type_ids = self.token_type_ids[:, :seq_length]
            buffered_token_type_ids_expanded = buffered_token_type_ids.expand(input_shape[0], seq_length)
            token_type_ids = buffered_token_type_ids_expanded
        
        inputs_embeddings = self.word_embeddings(input_ids)
        token_type_embeddings = self.token_type_embeddings(token_type_ids)
        position_embeddings = self.position_embeddings(position_ids)
        embeddings = inputs_embeddings + token_type_embeddings + position_embeddings

        embeddings = self.LayerNorm(embeddings)
        embeddings = self.dropout(embeddings)
        return embeddings

if __name__ == "__main__":
    # TODO: testing
    le = LearnableEmbeddings(512, 768, 0)
    print(le(LongTensor([1, 2, 3, 4, 5])))
    print(le(LongTensor([0, 0, 511, 510])))

    be = BERTEmbeddings()
    print(be(LongTensor([[1, 2, 3, 4, 5]])))
    print(be(LongTensor([[0, 1, 510, 511]])))
    