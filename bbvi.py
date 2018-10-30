import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.distributions.categorical as categorical


class WordVectorVariationalDistribution(nn.Module):

    def __init__(self, dim, length, embedding):
        super().__init__()
        self.wordvec_params = nn.Parameter(torch.empty(length, dim).uniform_())
        self.embedding = embedding

    def forward(self):
        logits = F.linear(self.wordvec_params, weight=self.embedding.weight)
        dist = categorical.Categorical(logits=logits)
        return dist.sample()

