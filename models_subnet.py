import torch.nn as nn
import torch.nn.functional as F

class TextSubnet(nn.Module):
    def __init__(self, in_dim, nb_classes, hidden_dims = [2,4,8]):
        super(TextSubnet, self).__init__()
        layers = []
        current_dim = in_dim

        # Dynamically create layers based on hidden_dims
        for factor in hidden_dims:
            next_dim = in_dim // factor
            layers.append(nn.Linear(current_dim, next_dim))
            layers.append(nn.LayerNorm(next_dim))
            layers.append(nn.ReLU(inplace=True))
            current_dim = next_dim
        
        # Add the final output layer
        layers.append(nn.Linear(current_dim, nb_classes))
        layers.append(nn.Sigmoid())

        # Combine all layers in a sequential container
        self.network = nn.Sequential(*layers)
        
        # Initialize weights
        self._initialize_weights()

    def _initialize_weights(self):
        # Initialize weights of linear layers with Xavier initialization
        for layer in self.network:
            if isinstance(layer, nn.Linear):
                nn.init.xavier_uniform_(layer.weight)
    
    def forward(self, x):
        return self.network(x)
