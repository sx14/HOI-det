import torch.nn as nn


class FCLayers(nn.Module):

    def __init__(self, in_feat_dim, class_num):
        super(FCLayers, self).__init__()
        self.feat_dim = in_feat_dim
        self.hidden_layer = nn.Sequential(
            nn.LeakyReLU(),
            nn.Dropout(p=0.5),
            nn.Linear(in_feat_dim, 1024))

        self.classifier = nn.Sequential(
            nn.LeakyReLU(),
            nn.Dropout(p=0.5),
            nn.Linear(1024, class_num))

    def forward(self, input):
        hidden = self.hidden_layer(input)
        classme = self.classifier(hidden)
        return classme






