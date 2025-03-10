import torch
import torch.nn as nn

class NeuralNetwork(nn.Module):
    def __init__(self,
            vocab_size,
            embedding_dim,
            n_input_tokens,
            d_model,
            n_layers,
            output_size,
            nonlinearity=nn.ReLU,
            lnorm=False,
            *args, **kwargs,
        ):
        """
        Args:
            vocab_size: int
                size of the vocabulary
            embedding_dim: int
                size of the word embeddings
            n_input_tokens: int
                number of input tokens in each independent
                input sequence. Essentially the sequence
                length.
            d_model: int
                size of the hidden layers
            n_layers: int
                number of hidden layers
            output_size: int
                size of the output layer
            nonlinearity: torch.nn.Module or str
                nonlinearity to use in the hidden layers
        """
        super(NeuralNetwork, self).__init__()
        if type(nonlinearity)==str:
            nonlinearity = getattr(torch.nn, nonlinearity)
        if n_layers < 1:
            raise ValueError("n_layers must be greater than 0")
        self.embedding = nn.Embedding(vocab_size, embedding_dim)
        inpt_dim = embedding_dim * n_input_tokens
        if n_layers == 1:
            self.hidden_layers = nn.Sequential(
                nn.Linear(inpt_dim, output_size)
            )
            return
        layers = []
        if lnorm:
            layers.append(nn.LayerNorm(inpt_dim))
        layers.append(
            nn.Linear(inpt_dim, d_model),
        )
        layers.append(nonlinearity())
        for _ in range(max(0,n_layers-2)):
            if lnorm:
                layers.append(nn.LayerNorm(d_model))
            layers.append(nn.Linear(d_model, d_model))
            layers.append(nonlinearity())
        if lnorm:
            layers.append(nn.LayerNorm(d_model))
        layers.append(nn.Linear(d_model, output_size))
        self.hidden_layers = nn.Sequential(*layers)

    def get_device(self):
        device = next(self.parameters()).get_device()
        return device if device>=0 else "cpu"

    def freeze_parameters(self):
        for p in self.parameters():
            try:
                p.requires_grad = False
            except: pass

    def forward(self, x, *args, **kwargs):
        x = self.embedding(x).reshape(x.size(0), -1)
        x = self.hidden_layers(x)
        return x

