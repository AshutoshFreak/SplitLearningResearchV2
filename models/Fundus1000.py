import torchvision
import torch.nn as nn

vgg_model = torchvision.models.vgg16(pretrained=True)
num_classes = 39
# print(vgg_model)

for param in vgg_model.parameters():
    param.requires_grad=False

vgg_model.classifier = nn.Sequential(
    nn.Linear(25088,2048),
    nn.ReLU(),
    nn.Dropout(p=0.37),
    nn.Linear(2048,1024),
    nn.ReLU(),
    nn.Dropout(p=0.5),
    nn.Linear(1024,num_classes)
)

# class Identity(nn.Module):
#     def __init__(self):
#         super().__init__()


#     def forward(self,x):
#         return x


class front(nn.Module):
    def __init__(self):
        super(front, self).__init__()
        self.front_m = vgg_model.features[:7]
    
    def forward(self, x):
        x = self.front_m(x)
        return x


class center(nn.Module):
    def __init__(self):
        super(center, self).__init__()
        self.center_m = vgg_model.features[7:]
        self.avgpool = vgg_model.avgpool

    def forward(self, x):
        x = self.center_m(x)
        x = self.avgpool(x)
        return x


class back(nn.Module):
    def __init__(self):
        super(back, self).__init__()
        self.back_m = vgg_model.classifier

    def forward(self, x):
        x = self.back_m(x)
        return x
