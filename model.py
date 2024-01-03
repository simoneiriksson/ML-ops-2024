from torch import nn

class MyAwesomeModel(nn.Module):
    """My awesome model."""

    def __init__(self):
        super().__init__()
        self.seq = nn.Sequential(
            nn.Flatten(),
            nn.Linear(28*28, 256),
            nn.ReLU(),
            nn.Linear(256, 10),
            nn.LogSoftmax(dim=1)
        )
    def forward(self, x):
        return self.seq(x)


class MyAwesomeModelConv(nn.Module):
    """My awesome model."""

    def __init__(self):
        super().__init__()
        self.seq = nn.Sequential(
            nn.Conv2d(1, 4, 3),  # 1x28x28 -> 4x26x26
            nn.ReLU(),
            nn.MaxPool2d(2, 2),  # 4x26x26 -> 4x13x13
            nn.Conv2d(4, 8, 3),  # 4x13x13 -> 8x11x11
            nn.ReLU(),
            nn.MaxPool2d(2, 2),  # 8x11x11 -> 8x5x5
            nn.Conv2d(8, 16, 3),  # 8x5x5 -> 16x3x3
            nn.ReLU(),
            nn.Flatten(),  # 16x3x3 -> 144
            nn.Linear(144, 10),
            nn.LogSoftmax(dim=1)
        )
    def forward(self, x):
        return self.seq(x)
