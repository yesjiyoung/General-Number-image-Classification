import torch.nn as nn


class ModelClass(nn.Module):

    def __init__(self):
        super(ModelClass, self).__init__()
        self.keep_prob = 0.5

        n_channels_1 = 6
        n_channels_2 = 16

        self.layer1 = nn.Sequential(
            nn.Conv2d(3, n_channels_1, kernel_size=3, stride=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
            # batchNorm2d 추가
            nn.BatchNorm2d(n_channels_1)
        )

        self.layer2 = nn.Sequential(
            nn.Conv2d(n_channels_1, n_channels_2, kernel_size=5, stride=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
            # batchNorm2d 추가
            nn.BatchNorm2d(n_channels_2)
        )

        self.fc3 = nn.Linear(4 * 4 * n_channels_2, 120, bias=True)
        nn.init.xavier_uniform_(self.fc3.weight)
        self.layer3 = nn.Sequential(
            self.fc3,
            nn.ReLU(),
            # batchNorm1d 추가
            nn.BatchNorm1d(120),
            nn.Dropout(p=1 - self.keep_prob)
        )

        self.fc4 = nn.Linear(120, 80, bias=True)
        nn.init.xavier_uniform_(self.fc4.weight)
        self.layer4 = nn.Sequential(
            self.fc4,
            nn.ReLU(),
            # batchNorm1d 추가
            nn.BatchNorm1d(80),
            nn.Dropout(p=1 - self.keep_prob)
        )

        self.fc5 = nn.Linear(80, 10, bias=True)
        # batchNorm1d 추가
        nn.BatchNorm1d(10)
        nn.init.xavier_uniform_(self.fc5.weight)

        # Softmax is included in nn.CrossEntropyLoss

    def forward(self, x):
        out = self.layer1(x)
        out = self.layer2(out)
        out = out.view(out.size(0), -1)  # Flatten them for FC

        out = self.layer3(out)
        out = self.layer4(out)

        out = self.fc5(out)
        return out