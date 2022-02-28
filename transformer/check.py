import random

import numpy as np
import torch

def create_model(SEED=42):

    random.seed(SEED)
    np.random.seed(SEED)
    torch.manual_seed(SEED)
    torch.cuda.manual_seed(SEED)

    from bert_model import BertModel

    our_bert = BertModel()

    random.seed(SEED)
    np.random.seed(SEED)
    torch.manual_seed(SEED)
    torch.cuda.manual_seed(SEED)

    from transformers import AutoConfig, AutoModel

    model_name = "bert-base-uncased"
    config = AutoConfig.from_pretrained(model_name)   # 'bart-base'
    model = AutoModel.from_config(config)

    return our_bert, model

def paramCheck(our_bert, hf_bert):
    print(our_bert.embeddings.word_embeddings.weight)
    print(hf_bert.embeddings.word_embeddings.weight)
    assert our_bert.embeddings.word_embeddings.weight == hf_bert.embeddings.word_embeddings.weight

if __name__ == "__main__":
    our_bert, hf_bert = create_model(SEED=42)
    paramCheck(our_bert, hf_bert)