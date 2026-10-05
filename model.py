import torch
import torch.nn as nn
from torch.nn import functional as F

def LN(x: torch.Tensor, eps=1e-5):
    mu = x.mean(dim=-1, keepdim=True)
    x = x - mu
    std = x.std(dim=-1, keepdim=True)
    x = x / (std + eps)
    return x, mu, std

class AutoEncoder(torch.Module):
    def __init__(
            self,
            input_dimension=768,
            latent_dimension=50000,
            activation=torch.topk,
            tied=False,
            normalize=False
    ):
        super().__init__()

        self.pre_bias = nn.Parameter(torch.zeros(input_dimension))
        self.encoder = nn.Linear(input_dimension, latent_dimension, bias=False)
        self.latent_bias = nn.Parameter(torch.zeros(latent_dimension))
        self.activation = activation

        if tied:
            self.decoder = TiedTranspose(self.encoder)
        else:
            self.decoder = nn.Linear(latent_dimension, input_dimension, bias=False)

        self.normalize = normalize

    # Accepts the input and also the slice of the latent dimension to return
    # Returns the latent representation before activation
    def encode_pre_act(self, x, latent_slice=slice(None)):
        x = x - self.pre_bias
        latents_pre_act = F.linear(
            x, self.encoder.weight[latent_slice], self.latent_bias[latent_slice]
        )
        return latents_pre_act

    # Returns a tuple of (normalized input, dict(mu=mean, std=std) or empty dict)
    def preprocess(self, x):
        if not self.normalize:
            return x, dict()
        
        x, mu, std = LN(x)
        return x, dict(mu=mu, std=std)

    # Returns a tuple of (latent representation, dict(mu=mean, std=std) or empty dict)
    def encode(self, x):
        x, info = self.preprocess(x)
        return self.activation(self.encode_pre_act(x)), info

    # Decodes and unnormalizes the latent representation
    def decode(self, latents, info):
        output = self.decoder(latents) + self.pre_bias

        if self.normalize:
            output = output * info["std"] + info["mu"]
        
        return output

    # Returns a tuple (latent representation before activation, latent representation, reconstructed output)
    def forward(self, x):
        x, info = self.preprocess(x)
        latents_pre_act = self.encode_pre_act(x)
        latents = self.activation(latents_pre_act)
        reconstructed_activation = self.decode(latents, info)

        return latents_pre_act, latents, reconstructed_activation

class TiedTranspose(nn.Module):
    def __init__(self, linear):
        super().__init__()
        self.linear = linear

    def forward(self, x):
        return F.linear(x, self.linear.weight.t(), None)

    @property
    def weight(self):
        return self.linear.weight.t()

    @property
    def bias(self):
        return self.linear.bias