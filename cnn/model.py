import torch.nn as nn
import torch
import torch.nn.functional as F

'''
class SimpleCNN(nn.Module):
    def __init__(self, num_classes):
        super(SimpleCNN, self).__init__()
        self.conv1 = nn.Conv2d(3, 32, kernel_size=5, stride=2)
        self.conv2 = nn.Conv2d(32, 32, kernel_size=5, stride=2)
        self.conv3 = nn.Conv2d(32, 64, kernel_size=5, stride=2)
        self.fc1 = nn.Linear(6*6*64, 256)
        self.fc2 = nn.Linear(256, num_classes)

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.dropout(x, p=0.5, training=self.training)
        x = F.relu(F.max_pool2d(self.conv2(x), 2))
        x = F.dropout(x, p=0.5, training=self.training)
        x = F.relu(F.max_pool2d(self.conv3(x), 2))
        x = F.dropout(x, p=0.5, training=self.training)
        #print(x.shape)

        x = x.view(-1,6*6*64)
        x = F.relu(self.fc1(x))
        x = F.dropout(x, training=self.training)
        x = self.fc2(x)
        return x
'''

class SimpleCNN(nn.Module):
    def __init__(self, num_classes, num_cov_layer, channels,
                 kernel_size, stride, num_mlp_layer, mlp_hids,
                 adp_output_size = 6, dropout = 0.5):
        super(SimpleCNN, self).__init__()
        num_channels = self.parse_layer_para(channels)
        num_mlp_hids = self.parse_layer_para(mlp_hids)

        assert num_cov_layer == len(num_channels) and num_mlp_layer == len(num_mlp_hids)

        modules = []
        for layer in range(num_cov_layer):
            if layer == 0:
                cov = nn.Conv2d(3, num_channels[layer], kernel_size= 11, stride = 4, padding= 2)
            else:
                cov = nn.Conv2d(num_channels[layer - 1], num_channels[layer],
                                kernel_size= kernel_size, padding= 1)
            modules.append(cov)
            modules.append(nn.ReLU(inplace=True))
            if layer < 2:
                modules.append(nn.MaxPool2d(kernel_size=kernel_size, stride=stride))
        if num_cov_layer > 2:
            modules.append(nn.MaxPool2d(kernel_size= kernel_size, stride=stride))

        self.features = nn.Sequential(*modules)

        self.avgpool = nn.AdaptiveAvgPool2d((adp_output_size, adp_output_size))

        classifier = []
        for layer in range(num_mlp_layer):
            classifier.append(nn.Dropout(p = dropout))
            if layer == 0:
                line = nn.Linear(num_channels[num_cov_layer - 1] * adp_output_size * adp_output_size, num_mlp_hids[layer])
            else:
                line = nn.Linear(num_mlp_hids[layer - 1], num_mlp_hids[layer])
            classifier.append(line)
            classifier.append(nn.ReLU(inplace= True))
        classifier.append(nn.Linear(num_mlp_hids[num_mlp_layer - 1], num_classes))

        self.classifier = nn.Sequential(*classifier)


    def forward(self, x):
        x = self.features(x)
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.classifier(x)
        return x

    def parse_layer_para(self, para_list_str):
        para_list_str = para_list_str.strip().split()
        return [int(para) for para in para_list_str]