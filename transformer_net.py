
from torch import nn
import torch
import numpy as np


K = 16
class Dense_Layer(nn.Module):
    def __init__(self,batch_number,loop):
        super(Dense_Layer, self).__init__()
        self.batch_number = batch_number
        self.BN = nn.InstanceNorm2d(K + K*batch_number)
        self.relu = nn.ReLU()
        self.conv = nn.Conv2d(loop*64 + K + K*batch_number, K, kernel_size = 3,stride = 1,padding = 1,bias=True)
        self.Dp = nn.Dropout(0.2)

    def forward(self, x):
        x = self.BN(x)
        x = self.relu(x)
        x = self.conv(x)
        x = self.Dp(x)
        return x

class Transition_Down(nn.Module):
    def __init__(self,in_channels):
        super(Transition_Down, self).__init__()
        self.TD = nn.Sequential(nn.InstanceNorm2d(in_channels),
                                nn.ReLU(),
                                nn.Conv2d(in_channels,in_channels,kernel_size=1),
                                nn.Dropout(0.2),
                                nn.MaxPool2d(2),)
    def forward(self, x):
        x = self.TD(x)
        return x

class Transition_Up(nn.Module):
    def __init__(self,in_channels):
        super(Transition_Up, self).__init__()
        self.TU = nn.Sequential(nn.Upsample(mode='nearest',scale_factor = 2),
                                nn.ReflectionPad2d(1),
                                nn.Conv2d(in_channels,in_channels,kernel_size=3,stride=1)
                                )

    def forward(self, x):
        x = self.TU(x)
        return x

class TransformerNet(nn.Module):
    def __init__(self,):
        super(TransformerNet,self).__init__()
        self.start = nn.Conv2d(in_channels = 3,out_channels = K,kernel_size=3,stride=1,padding=1 )
        self.Dense_layer1_0 = Dense_Layer(0, 0)  #input k output k
        self.Dense_layer1_1 = Dense_Layer(1, 0)  #input 2k output k
        self.Dense_layer1_2 = Dense_Layer(2, 0)  #input 3k output k
        self.Dense_layer1_3 = Dense_Layer(3, 0)  #input 4k output k
        self.TD1 = Transition_Down(K + K * 4) #input 5k output 5k
        self.Dense_layer2_0 = Dense_Layer(0, 1)  #input 5k output k
        self.Dense_layer2_1 = Dense_Layer(1, 1)  #input 6k output k
        self.Dense_layer2_2 = Dense_Layer(2, 1)  #input 7k output k
        self.Dense_layer2_3 = Dense_Layer(3, 1)  #input 8k output k
        self.TD2 = Transition_Down(K + K * 8)  # input 9k output 9k
        self.Dense_layer3_0 = Dense_Layer(0, 2)  #input 9k output k
        self.Dense_layer3_1 = Dense_Layer(1, 2)  #input 10k output k
        self.Dense_layer3_2 = Dense_Layer(2, 2)  #input 11k output k
        self.Dense_layer3_3 = Dense_Layer(3, 2)  #input 12k output k
        self.TU1 = Transition_Up(K + K * 3)  # input 4k output 4k

        self.Dense_layer4_0 = Dense_Layer(3, 0)  #input 4k output k
        self.Dense_layer4_1 = Dense_Layer(4, 0)  #input 5k output k
        self.Dense_layer4_2 = Dense_Layer(5, 0)  #input 6k output k
        self.Dense_layer4_3 = Dense_Layer(6, 0)  #input 7k output k
        self.TU2 = Transition_Up(K + K * 3)  # input 4k output 4k


        self.Dense_layer5_0 = Dense_Layer(3, 0)  #input 4k output k
        self.Dense_layer5_1 = Dense_Layer(4, 0)  #input 5k output k
        self.Dense_layer5_2 = Dense_Layer(5, 0)  #input 6k output k
        self.Dense_layer5_3 = Dense_Layer(6, 0)  #input 7k output k

        self.over = nn.Conv2d(in_channels=4 * K,out_channels=3,kernel_size=1)

    def forward(self, X):
        X = self.start(X)  #size = 16,256,256
        x1 = self.Dense_layer1_0(X) #16,256,256
        cat_data0 = torch.cat([x1,X],dim = 1)  #32,256,256
        x2 = self.Dense_layer1_1(cat_data0) #16,256,256
        cat_data1 = torch.cat([x2,cat_data0],dim = 1) #48,256,256
        x3 = self.Dense_layer1_2(cat_data1)
        cat_data2 = torch.cat([x3,cat_data1],dim = 1) #64.256.256
        x4 = self.Dense_layer1_3(cat_data2)
        output_layer1 = torch.cat([x1,x2,x3,x4],dim = 1) #64,256,256
        output_layer1_cat = torch.cat([output_layer1,X],dim = 1) #5k,256,256
        TD1 = self.TD1(output_layer1_cat)#5k,128,128
        ##########################################################################
        x1 = self.Dense_layer2_0(TD1)
        cat_data0 = torch.cat([x1,TD1],dim = 1)
        x2 = self.Dense_layer2_1(cat_data0)
        cat_data1 = torch.cat([x2,cat_data0],dim = 1)
        x3 = self.Dense_layer2_2(cat_data1)
        cat_data2 = torch.cat([x3,cat_data1],dim = 1)
        x4 = self.Dense_layer2_3(cat_data2)
        output_layer1 = torch.cat([x1,x2,x3,x4],dim = 1)
        output_layer2_cat = torch.cat([output_layer1,TD1],dim = 1) #9k,128,128
        TD2 = self.TD2(output_layer2_cat)#9k,64,64
        ##########################################################################
        x1 = self.Dense_layer3_0(TD2)
        cat_data0 = torch.cat([x1,TD2],dim = 1)
        x2 = self.Dense_layer3_1(cat_data0)
        cat_data1 = torch.cat([x2,cat_data0],dim = 1)
        x3 = self.Dense_layer3_2(cat_data1)
        cat_data2 = torch.cat([x3,cat_data1],dim = 1)
        x4 = self.Dense_layer3_3(cat_data2)
        output_layer1 = torch.cat([x1,x2,x3,x4],dim = 1) #4k,64,64
        TU1 = self.TU1(output_layer1) #4k,128,128

        ##########################################################################
        x1 = self.Dense_layer4_0(TU1)
        cat_data0 = torch.cat([x1,TU1],dim = 1)
        x2 = self.Dense_layer4_1(cat_data0)
        cat_data1 = torch.cat([x2,cat_data0],dim = 1)
        x3 = self.Dense_layer4_2(cat_data1)
        cat_data2 = torch.cat([x3,cat_data1],dim = 1)
        x4 = self.Dense_layer4_3(cat_data2)
        output_layer1 = torch.cat([x1,x2,x3,x4],dim = 1) #4k,128,128
        TU2 = self.TU2(output_layer1)#4k,256,256

        ##########################################################################
        x1 = self.Dense_layer5_0(TU2)
        cat_data0 = torch.cat([x1,TU2],dim = 1)
        x2 = self.Dense_layer5_1(cat_data0)
        cat_data1 = torch.cat([x2,cat_data0],dim = 1)
        x3 = self.Dense_layer5_2(cat_data1)
        cat_data2 = torch.cat([x3,cat_data1],dim = 1)
        x4 = self.Dense_layer5_3(cat_data2)
        output_layer1 = torch.cat([x1,x2,x3,x4],dim = 1) #4k,256,256
        output = self.over(output_layer1)
        return output
