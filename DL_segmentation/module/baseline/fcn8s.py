import torch.nn as nn
import torch.nn.functional as F
from torchvision.models.vgg import vgg16
from torchvision import models
from ever.interface import ERModule
from ever import registry
import torch
from module.loss import SegmentationLoss
from collections import OrderedDict

@registry.MODEL.register()
class FCN8s(ERModule):
    def __init__(self, config):
        super().__init__(config)
        # self.aux = aux
        self.loss = SegmentationLoss(self.config.loss)
        #self.pretrained = vgg16(pretrained=self.config.pretrained).features
        base_model = models.resnet50(pretrained=True)
        old_state_dict = base_model.state_dict()
        base_model.fc = nn.Linear(2048, 512)
        
        checkpoint = torch.load('/home/mz2466/checkpoint_0030.pth.tar')
        print("***** Loaded checkpoint_0030.pth.tar *****")
        state_dict = checkpoint['state_dict']
        new_state_dict = OrderedDict()
        
        for k, v in state_dict.items():
            if 'encoder_k' in k:
                continue
            if 'module' in k:
                k = k.replace('module.', '')
            if 'encoder_q' in k:
                k = k.replace('encoder_q.', '')
            if 'fc.2.weight' in k:
                k = k.replace('fc.2.weight', 'fc.weight')
                v = old_state_dict['fc.weight']
            if 'fc.2.bias' in k:
                k = k.replace('fc.2.bias', 'fc.bias')
                v = old_state_dict['fc.bias']
            if (k in ["queue", "queue_ptr", "fc.0.weight", "fc.0.bias"]):
                continue
            new_state_dict[k]=v
        base_model.load_state_dict(new_state_dict, strict=True)
        
        self.pretrained = base_model

        #self.pool3 = nn.Sequential(*self.pretrained[:17])
        #self.pool4 = nn.Sequential(*self.pretrained[17:24])
        #self.pool5 = nn.Sequential(*self.pretrained[24:])
        self.head = _FCNHead(512, self.config.classes, nn.BatchNorm2d)
        self.score_pool3 = nn.Conv2d(256, self.config.classes, 1)
        self.score_pool4 = nn.Conv2d(512, self.config.classes, 1)
        # if aux:
        #     self.auxlayer = _FCNHead(512, nclass, norm_layer)


    def forward(self, x, y=None):
        #pool3 = self.pool3(x)
        #pool4 = self.pool4(pool3)
        #pool5 = self.pool5(pool4)

        pool5 = self.pretrained(x)

        # outputs = []
        score_fr = self.head(pool5)

        score_pool4 = self.score_pool4(pool4)
        score_pool3 = self.score_pool3(pool3)

        upscore2 = F.interpolate(score_fr, score_pool4.size()[2:], mode='bilinear', align_corners=True)
        fuse_pool4 = upscore2 + score_pool4

        upscore_pool4 = F.interpolate(fuse_pool4, score_pool3.size()[2:], mode='bilinear', align_corners=True)
        fuse_pool3 = upscore_pool4 + score_pool3

        cls_pred = F.interpolate(fuse_pool3, x.size()[2:], mode='bilinear', align_corners=True)


        if self.training:
            return self.loss(cls_pred, y['cls'])

        return cls_pred.softmax(dim=1)



    def set_default_config(self):
        self.config.update(dict(
            classes=7,
            pretrained=True,
            loss=dict(
                ignore_index=-1
            ),
        ))

class _FCNHead(nn.Module):
    def __init__(self, in_channels, channels, norm_layer=nn.BatchNorm2d, **kwargs):
        super(_FCNHead, self).__init__()
        inter_channels = in_channels // 4
        self.block = nn.Sequential(
            nn.Conv2d(in_channels, inter_channels, 3, padding=1, bias=False),
            norm_layer(inter_channels),
            nn.ReLU(inplace=True),
            nn.Dropout(0.1),
            nn.Conv2d(inter_channels, channels, 1)
        )

    def forward(self, x):
        return self.block(x)



if __name__ == '__main__':
    model = FCN8s(dict())
    x = torch.ones(2, 3, 512, 512)
    y = torch.ones(2, 512, 512)
    print(model(x, dict(cls=y.long())))
