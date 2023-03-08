import torch
import torch.nn as nn
import torchvision
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from torchvision.models.detection.mask_rcnn import MaskRCNNPredictor


class Block(nn.Module):
    def __init__(self, in_channels, out_channels, kernel = 4, stride=2, down=True, act="relu", use_dropout=False):
        super(Block, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel, stride, 1, bias=False, padding_mode="reflect")
            if down
            else nn.ConvTranspose2d(in_channels, out_channels, kernel, stride, 1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU() if act == "relu" else nn.LeakyReLU(0.2)
        )
        self.use_dropout = use_dropout
        self.dropout = nn.Dropout(0.5)
        self.down = down

    def forward(self, x):
        x = self.conv(x)
        return self.dropout(x) if self.use_dropout else x


class Unet128(nn.Module):
    
    def __init__(self, in_channels = 3, out_channels=3, features=64):
        super(Unet128,self).__init__()
        
        self.initial_down = nn.Sequential(
            nn.Conv2d(in_channels, features, 4, 2, 1, padding_mode="reflect"),
            nn.LeakyReLU(0.2)
        )
        
        self.down1 = Block(features, features*2, down=True, act="leaky", use_dropout=False)
        self.down2 = Block(
            features*2, features * 4, down=True, act="leaky", use_dropout=False
        )
        self.down3 = Block(features * 4, features * 8, down=True, act="leaky", use_dropout=False)
        self.down4 = Block(features * 8, features * 8, down=True, act="leaky", use_dropout=False)
        self.down5 = Block(features * 8, features * 8, down=True, act="leaky", use_dropout=False)
        
        self.bottleneck = nn.Sequential(
            nn.Conv2d(features * 8, features * 8, 4, 2, 1, padding_mode="reflect"),
            nn.ReLU()
        )
        self.up1 = Block(features * 8, features * 8, down=False, act="relu", use_dropout=True)
        self.up2 = Block(features * 8 * 2, features * 8, down=False, act="relu", use_dropout=True)
        self.up3 = Block(features * 8 * 2, features * 8, down=False, act="relu", use_dropout=True)
        self.up4 = Block(features * 8 * 2, features * 4, down=False, act="relu", use_dropout=False)
        self.up5 = Block(features * 4 * 2, features *2, down=False, act="relu", use_dropout=False)
        self.up6 = Block(features *2*2, features , down=False, act="relu", use_dropout=False)
        
        self.final_up = nn.Sequential(
            nn.ConvTranspose2d(features*2, out_channels, kernel_size=4, stride=2, padding=1),
            nn.Tanh(),
        )
        
    def forward(self, x):

        d1 = self.initial_down(x)
        d2 = self.down1(d1)
        d3 = self.down2(d2)
        d4 = self.down3(d3)
        d5 = self.down4(d4)
        d6 = self.down5(d5)
        #print(d6.shape)
        bottleneck = self.bottleneck(d6)

        up1 = self.up1(bottleneck)
        #print('up1',up1.shape, d6.shape)
        up2 = self.up2(torch.cat([up1, d6], 1))
        #print('up2',up2.shape, d5.shape)
        up3 = self.up3(torch.cat([up2, d5], 1))
        #print('up3',up3.shape, d4.shape)
        up4 = self.up4(torch.cat([up3, d4], 1))
        #print('up4',up4.shape, d3.shape)
        up5 = self.up5(torch.cat([up4, d3], 1))
        #print('up5',up5.shape, d2.shape)
        up6 = self.up6(torch.cat([up5, d2], 1))
        #print('up6',up6.shape, d1.shape)
        upfinal  = self.final_up(torch.cat([up6, d1], 1))
        
        return upfinal    



class Unet256(nn.Module):
    
    def __init__(self, in_channels = 3, out_channels=3, features=64):
        super(Unet256,self).__init__()
        
        self.initial_down = nn.Sequential(
            nn.Conv2d(in_channels, features, 4, 2, 1, padding_mode="reflect"),
            nn.LeakyReLU(0.2)
        )
        
        self.down1 = Block(features, features*2, down=True, act="leaky", use_dropout=False)
        self.down2 = Block(
            features*2, features * 4, down=True, act="leaky", use_dropout=False
        )
        self.down3 = Block(features * 4, features * 8, down=True, act="leaky", use_dropout=False)
        self.down4 = Block(features * 8, features * 8, down=True, act="leaky", use_dropout=False)
        self.down5 = Block(features * 8, features * 8, down=True, act="leaky", use_dropout=False)
        self.down6 = Block(features * 8, features * 8, down=True, act="leaky", use_dropout=False)
        
        self.bottleneck = nn.Sequential(
            nn.Conv2d(features * 8, features * 8, 4, 2, 1, padding_mode="reflect"),
            nn.ReLU()
        )
        self.up1 = Block(features * 8, features * 8, down=False, act="relu", use_dropout=True)
        
        self.up2 = Block(features * 8 * 2, features * 8, down=False, act="relu", use_dropout=True)
        self.up3 = Block(features * 8 * 2, features * 8, down=False, act="relu", use_dropout=True)
        self.up4 = Block(features * 8 * 2, features * 8, down=False, act="relu", use_dropout=False)
        self.up5 = Block(features * 8 * 2, features *4, down=False, act="relu", use_dropout=False)
        self.up6 = Block(features *4*2, features*2 , down=False, act="relu", use_dropout=False)
        self.up7 = Block(features*2*2, features, down=False, act="relu", use_dropout=False)

        self.final_up = nn.Sequential(
            nn.ConvTranspose2d(features*2, out_channels, kernel_size=4, stride=2, padding=1),
            nn.Sigmoid(),
        )

    def forward(self, x):

        d1 = self.initial_down(x)
        d2 = self.down1(d1)
        d3 = self.down2(d2)
        d4 = self.down3(d3)
        d5 = self.down4(d4)
        d6 = self.down5(d5)
        d7 = self.down6(d6)
        bottleneck = self.bottleneck(d7)

        up1 = self.up1(bottleneck)
        
        up2 = self.up2(torch.cat([up1, d7], 1))
        #print('up2',up2.shape, d6.shape)
        up3 = self.up3(torch.cat([up2, d6], 1))
        #print('up3',up3.shape, d5.shape)
        up4 = self.up4(torch.cat([up3, d5], 1))
        #print('up4',up4.shape, d4.shape)
        up5 = self.up5(torch.cat([up4, d4], 1))
        #print('up5',up5.shape, d3.shape)
        up6 = self.up6(torch.cat([up5, d3], 1))
        #print('up6',up6.shape, d2.shape)
        up7 = self.up7(torch.cat([up6, d2], 1))
        #print('up7',up7.shape, d1.shape)
        upfinal  = self.final_up(torch.cat([up7, d1], 1))
        
        return upfinal
    

def get_instance_segmentation_model(num_classes):
    # load an instance segmentation model pre-trained on COCO
    model = torchvision.models.detection.maskrcnn_resnet50_fpn(pretrained=True)

    # get the number of input features for the classifier
    in_features = model.roi_heads.box_predictor.cls_score.in_features
    # replace the pre-trained head with a new one
    model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)

    # now get the number of input features for the mask classifier
    in_features_mask = model.roi_heads.mask_predictor.conv5_mask.in_channels
    hidden_layer = 256
    # and replace the mask predictor with a new one
    model.roi_heads.mask_predictor = MaskRCNNPredictor(in_features_mask,
                                                       hidden_layer,
                                                       num_classes)

    return model
