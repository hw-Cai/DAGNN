import torch
import torch.nn as nn


class Deep_AutoEncoder(torch.nn.Module):
    def __init__(self):
        super(Deep_AutoEncoder, self).__init__()

        ###加入代码###
        self.encoder = nn.Sequential(
            # ########修改的encoder 结构########
            nn.Linear(256, 128),
            nn.Tanh(),
            nn.Linear(128, 4),
            nn.Tanh()

        )
        self.decoder = nn.Sequential(
            nn.Linear(4, 128),
            nn.Tanh(),
            nn.Linear(128, 256)
        )

        # output
        self.output = nn.Sequential(
            nn.Tanh(),
            nn.Linear(256, 1),
            nn.Sigmoid()
        )

    def forward(self, data):
        self.x = data.x

        out = self.encoder(self.x)
        out = self.decoder(out)
        out = torch.squeeze(out, 1)
        return out
