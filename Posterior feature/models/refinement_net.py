import torch.nn as nn

# Arctecture of the refinement network
class RefinementNet(nn.Module):
    def __init__(self, feat_dim):
        super().__init__()

        self.fc = nn.Sequential(
            nn.Linear(feat_dim, 64),
            nn.ReLU(),

            nn.Linear(64, 32),
            nn.ReLU(),

            nn.Linear(32, 5),
        )

    def forward(self, x):
        x = self.fc(x)
        return x