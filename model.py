import torch.nn as nn

class EpigeneModel(nn.Module):
    def __init__(self):
        super().__init__()

        self.linear1 = nn.Linear(64, 128)
        self.linear2 = nn.Linear(128, 64)
        self.linear3 = nn.Linear(64, 32)
        self.linear4 = nn.Linear(32, 1)

        self.activation = nn.GELU()

        self.norm1 = nn.BatchNorm1d(128)
        self.norm2 = nn.BatchNorm1d(64)
        self.norm3 = nn.BatchNorm1d(32)

        self.dropout_input = nn.Dropout(p = 0.1)
        self.dropout_hidden = nn.Dropout(p = 0.3)

    def forward(self, x):
        # first layer
        x = self.dropout_input(x)
        x = self.linear1(x)
        x = self.norm1(x)
        x = self.activation(x)

        # second layer
        x = self.dropout_hidden(x)
        x = self.linear2(x)
        x = self.norm2(x)
        x = self.activation(x)

        # third layer
        x = self.dropout_hidden(x)
        x = self.linear3(x)
        x = self.norm3(x)
        x = self.activation(x)

        # output layer
        x = self.linear4(x)
        return x
