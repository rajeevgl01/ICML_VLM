import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoModel

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

class TextBackbone(nn.Module):
    def __init__(self, in_dim, out_dim, max_length, num_classes, is_classifier=False):
        super(TextBackbone, self).__init__()
        self.embedding_model = AutoModel.from_pretrained("nvidia/NV-Embed-v2", trust_remote_code=True)
        self.adapter = Adapter(in_dim, out_dim)
        self.norm = nn.LayerNorm(in_dim)
        self.is_classifier = is_classifier
        if self.is_classifier:
            self.head_norm = nn.LayerNorm(out_dim)
            self.head = nn.Linear(out_dim, num_classes)
            self._initialize_weights()

        self.max_length = max_length

        self.embedding_model.eval()

    def _initialize_weights(self):
        nn.init.xavier_uniform_(self.head.weight)

    def forward_head(self, x):
        x = self.head(self.head_norm(x))

    def forward(self, x):
        x = self.embedding_model.encode(x, instruction="", max_length=self.max_length)
        x = self.norm(x)
        features = self.adapter(x)

        if self.is_classifier:
            return self.forward_head(features), features
        return features