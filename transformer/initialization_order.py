import torch
from torch import nn

from embeddings import BERTEmbeddings

class Model1(nn.Module):

    def __init__(self, in_features = 768, out_features = 3072):
        super().__init__()
        self.embeddings = BERTEmbeddings()
        self.linear1 = nn.Linear(in_features, out_features)
        self.linear2 = nn.Linear(out_features, in_features)

class Model2(nn.Module):

    def __init__(self, in_features = 768, out_features = 3072):
        super().__init__()
        self.linear1 = nn.Linear(in_features, out_features)
        self.embeddings = BERTEmbeddings()
        # self.linear2 = nn.Linear(out_features, in_features)

SEED = 42
torch.manual_seed(SEED)
torch.cuda.manual_seed(SEED)
model1 = Model1()

torch.manual_seed(SEED)
torch.cuda.manual_seed(SEED)
model2 = Model2()

print(model1.linear1.weight)
print(model2.linear1.weight)

print(model1.embeddings.word_embeddings.weight)
print(model2.embeddings.word_embeddings.weight)

assert model1.linear1.weight == model2.linear1.weight