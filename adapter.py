import torch
import torch.nn as nn
import torch.nn.functional as F

class Adapter(nn.Module):
    def __init__(self, in_dim, out_dim):
        super(Adapter, self).__init__()
        self.fc1 = nn.Linear(in_dim, in_dim//2)
        self.fc2 = nn.Linear(in_dim//2, in_dim//4)
        self.fc3 = nn.Linear(in_dim//4, out_dim)
        self.relu = nn.ReLU(inplace=True)
    
        self._initialize_weights()

    def _initialize_weights(self):
        nn.init.xavier_uniform_(self.fc1.weight)
        nn.init.xavier_uniform_(self.fc2.weight)
        nn.init.xavier_uniform_(self.fc3.weight)
    
    def forward(self, x):
        x = self.relu(self.fc1(x))
        x = self.relu(self.fc2(x))
        x = self.relu(self.fc3(x))
        return x

class AdapterFeatures(nn.Module):
    def __init__(self, in_dim, out_dim):
        super(AdapterFeatures, self).__init__()
        self.adapter = Adapter(in_dim, out_dim)
        self.norm = nn.LayerNorm(out_dim)
    
    def forward(self, x):
        x = self.adapter(x)
        return self.norm(x)
    
class AdapterClassifier(nn.Module):
    def __init__(self, in_dim, out_dim, num_classes):
        super(AdapterClassifier, self).__init__()
        self.adapter = Adapter(in_dim, out_dim)
        self.norm = nn.LayerNorm(out_dim)
        self.head = nn.Linear(out_dim, num_classes)

        self._initialize_weights()

    def _initialize_weights(self):
        nn.init.xavier_uniform_(self.head.weight)
    
    def forward(self, x):
        x = self.adapter(x)
        features = self.norm(x)
        logits = self.head(features)
        return logits, features