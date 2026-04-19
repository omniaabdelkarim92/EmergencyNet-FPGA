# GENETARED BY NNDCT, DO NOT EDIT!

import torch
import pytorch_nndct as py_nndct
class SqueezeNet(torch.nn.Module):
    def __init__(self):
        super(SqueezeNet, self).__init__()
        self.module_0 = py_nndct.nn.Input() #SqueezeNet::input_0
        self.module_1 = py_nndct.nn.Conv2d(in_channels=3, out_channels=64, kernel_size=[3, 3], stride=[2, 2], padding=[0, 0], dilation=[1, 1], groups=1, bias=True) #SqueezeNet::SqueezeNet/Sequential[features]/Conv2d[0]/input.3
        self.module_2 = py_nndct.nn.ReLU(inplace=True) #SqueezeNet::SqueezeNet/Sequential[features]/ReLU[1]/2608
        self.module_3 = py_nndct.nn.MaxPool2d(kernel_size=[3, 3], stride=[2, 2], padding=[0, 0], dilation=[1, 1], ceil_mode=True) #SqueezeNet::SqueezeNet/Sequential[features]/MaxPool2d[2]/input.5
        self.module_4 = py_nndct.nn.Conv2d(in_channels=64, out_channels=16, kernel_size=[1, 1], stride=[1, 1], padding=[0, 0], dilation=[1, 1], groups=1, bias=True) #SqueezeNet::SqueezeNet/Sequential[features]/Fire[3]/Conv2d[squeeze]/input.7
        self.module_5 = py_nndct.nn.ReLU(inplace=True) #SqueezeNet::SqueezeNet/Sequential[features]/Fire[3]/ReLU[squeeze_activation]/input.9
        self.module_6 = py_nndct.nn.Conv2d(in_channels=16, out_channels=64, kernel_size=[1, 1], stride=[1, 1], padding=[0, 0], dilation=[1, 1], groups=1, bias=True) #SqueezeNet::SqueezeNet/Sequential[features]/Fire[3]/Conv2d[expand1x1]/input.11
        self.module_7 = py_nndct.nn.ReLU(inplace=True) #SqueezeNet::SqueezeNet/Sequential[features]/Fire[3]/ReLU[expand1x1_activation]/2662
        self.module_8 = py_nndct.nn.Conv2d(in_channels=16, out_channels=64, kernel_size=[3, 3], stride=[1, 1], padding=[1, 1], dilation=[1, 1], groups=1, bias=True) #SqueezeNet::SqueezeNet/Sequential[features]/Fire[3]/Conv2d[expand3x3]/input.13
        self.module_9 = py_nndct.nn.ReLU(inplace=True) #SqueezeNet::SqueezeNet/Sequential[features]/Fire[3]/ReLU[expand3x3_activation]/2682
        self.module_10 = py_nndct.nn.Cat() #SqueezeNet::SqueezeNet/Sequential[features]/Fire[3]/input.15
        self.module_11 = py_nndct.nn.Conv2d(in_channels=128, out_channels=16, kernel_size=[1, 1], stride=[1, 1], padding=[0, 0], dilation=[1, 1], groups=1, bias=True) #SqueezeNet::SqueezeNet/Sequential[features]/Fire[4]/Conv2d[squeeze]/input.17
        self.module_12 = py_nndct.nn.ReLU(inplace=True) #SqueezeNet::SqueezeNet/Sequential[features]/Fire[4]/ReLU[squeeze_activation]/input.19
        self.module_13 = py_nndct.nn.Conv2d(in_channels=16, out_channels=64, kernel_size=[1, 1], stride=[1, 1], padding=[0, 0], dilation=[1, 1], groups=1, bias=True) #SqueezeNet::SqueezeNet/Sequential[features]/Fire[4]/Conv2d[expand1x1]/input.21
        self.module_14 = py_nndct.nn.ReLU(inplace=True) #SqueezeNet::SqueezeNet/Sequential[features]/Fire[4]/ReLU[expand1x1_activation]/2725
        self.module_15 = py_nndct.nn.Conv2d(in_channels=16, out_channels=64, kernel_size=[3, 3], stride=[1, 1], padding=[1, 1], dilation=[1, 1], groups=1, bias=True) #SqueezeNet::SqueezeNet/Sequential[features]/Fire[4]/Conv2d[expand3x3]/input.23
        self.module_16 = py_nndct.nn.ReLU(inplace=True) #SqueezeNet::SqueezeNet/Sequential[features]/Fire[4]/ReLU[expand3x3_activation]/2745
        self.module_17 = py_nndct.nn.Cat() #SqueezeNet::SqueezeNet/Sequential[features]/Fire[4]/2748
        self.module_18 = py_nndct.nn.MaxPool2d(kernel_size=[3, 3], stride=[2, 2], padding=[0, 0], dilation=[1, 1], ceil_mode=True) #SqueezeNet::SqueezeNet/Sequential[features]/MaxPool2d[5]/input.25
        self.module_19 = py_nndct.nn.Conv2d(in_channels=128, out_channels=32, kernel_size=[1, 1], stride=[1, 1], padding=[0, 0], dilation=[1, 1], groups=1, bias=True) #SqueezeNet::SqueezeNet/Sequential[features]/Fire[6]/Conv2d[squeeze]/input.27
        self.module_20 = py_nndct.nn.ReLU(inplace=True) #SqueezeNet::SqueezeNet/Sequential[features]/Fire[6]/ReLU[squeeze_activation]/input.29
        self.module_21 = py_nndct.nn.Conv2d(in_channels=32, out_channels=128, kernel_size=[1, 1], stride=[1, 1], padding=[0, 0], dilation=[1, 1], groups=1, bias=True) #SqueezeNet::SqueezeNet/Sequential[features]/Fire[6]/Conv2d[expand1x1]/input.31
        self.module_22 = py_nndct.nn.ReLU(inplace=True) #SqueezeNet::SqueezeNet/Sequential[features]/Fire[6]/ReLU[expand1x1_activation]/2802
        self.module_23 = py_nndct.nn.Conv2d(in_channels=32, out_channels=128, kernel_size=[3, 3], stride=[1, 1], padding=[1, 1], dilation=[1, 1], groups=1, bias=True) #SqueezeNet::SqueezeNet/Sequential[features]/Fire[6]/Conv2d[expand3x3]/input.33
        self.module_24 = py_nndct.nn.ReLU(inplace=True) #SqueezeNet::SqueezeNet/Sequential[features]/Fire[6]/ReLU[expand3x3_activation]/2822
        self.module_25 = py_nndct.nn.Cat() #SqueezeNet::SqueezeNet/Sequential[features]/Fire[6]/input.35
        self.module_26 = py_nndct.nn.Conv2d(in_channels=256, out_channels=32, kernel_size=[1, 1], stride=[1, 1], padding=[0, 0], dilation=[1, 1], groups=1, bias=True) #SqueezeNet::SqueezeNet/Sequential[features]/Fire[7]/Conv2d[squeeze]/input.37
        self.module_27 = py_nndct.nn.ReLU(inplace=True) #SqueezeNet::SqueezeNet/Sequential[features]/Fire[7]/ReLU[squeeze_activation]/input.39
        self.module_28 = py_nndct.nn.Conv2d(in_channels=32, out_channels=128, kernel_size=[1, 1], stride=[1, 1], padding=[0, 0], dilation=[1, 1], groups=1, bias=True) #SqueezeNet::SqueezeNet/Sequential[features]/Fire[7]/Conv2d[expand1x1]/input.41
        self.module_29 = py_nndct.nn.ReLU(inplace=True) #SqueezeNet::SqueezeNet/Sequential[features]/Fire[7]/ReLU[expand1x1_activation]/2865
        self.module_30 = py_nndct.nn.Conv2d(in_channels=32, out_channels=128, kernel_size=[3, 3], stride=[1, 1], padding=[1, 1], dilation=[1, 1], groups=1, bias=True) #SqueezeNet::SqueezeNet/Sequential[features]/Fire[7]/Conv2d[expand3x3]/input.43
        self.module_31 = py_nndct.nn.ReLU(inplace=True) #SqueezeNet::SqueezeNet/Sequential[features]/Fire[7]/ReLU[expand3x3_activation]/2885
        self.module_32 = py_nndct.nn.Cat() #SqueezeNet::SqueezeNet/Sequential[features]/Fire[7]/2888
        self.module_33 = py_nndct.nn.MaxPool2d(kernel_size=[3, 3], stride=[2, 2], padding=[0, 0], dilation=[1, 1], ceil_mode=True) #SqueezeNet::SqueezeNet/Sequential[features]/MaxPool2d[8]/input.45
        self.module_34 = py_nndct.nn.Conv2d(in_channels=256, out_channels=48, kernel_size=[1, 1], stride=[1, 1], padding=[0, 0], dilation=[1, 1], groups=1, bias=True) #SqueezeNet::SqueezeNet/Sequential[features]/Fire[9]/Conv2d[squeeze]/input.47
        self.module_35 = py_nndct.nn.ReLU(inplace=True) #SqueezeNet::SqueezeNet/Sequential[features]/Fire[9]/ReLU[squeeze_activation]/input.49
        self.module_36 = py_nndct.nn.Conv2d(in_channels=48, out_channels=192, kernel_size=[1, 1], stride=[1, 1], padding=[0, 0], dilation=[1, 1], groups=1, bias=True) #SqueezeNet::SqueezeNet/Sequential[features]/Fire[9]/Conv2d[expand1x1]/input.51
        self.module_37 = py_nndct.nn.ReLU(inplace=True) #SqueezeNet::SqueezeNet/Sequential[features]/Fire[9]/ReLU[expand1x1_activation]/2942
        self.module_38 = py_nndct.nn.Conv2d(in_channels=48, out_channels=192, kernel_size=[3, 3], stride=[1, 1], padding=[1, 1], dilation=[1, 1], groups=1, bias=True) #SqueezeNet::SqueezeNet/Sequential[features]/Fire[9]/Conv2d[expand3x3]/input.53
        self.module_39 = py_nndct.nn.ReLU(inplace=True) #SqueezeNet::SqueezeNet/Sequential[features]/Fire[9]/ReLU[expand3x3_activation]/2962
        self.module_40 = py_nndct.nn.Cat() #SqueezeNet::SqueezeNet/Sequential[features]/Fire[9]/input.55
        self.module_41 = py_nndct.nn.Conv2d(in_channels=384, out_channels=48, kernel_size=[1, 1], stride=[1, 1], padding=[0, 0], dilation=[1, 1], groups=1, bias=True) #SqueezeNet::SqueezeNet/Sequential[features]/Fire[10]/Conv2d[squeeze]/input.57
        self.module_42 = py_nndct.nn.ReLU(inplace=True) #SqueezeNet::SqueezeNet/Sequential[features]/Fire[10]/ReLU[squeeze_activation]/input.59
        self.module_43 = py_nndct.nn.Conv2d(in_channels=48, out_channels=192, kernel_size=[1, 1], stride=[1, 1], padding=[0, 0], dilation=[1, 1], groups=1, bias=True) #SqueezeNet::SqueezeNet/Sequential[features]/Fire[10]/Conv2d[expand1x1]/input.61
        self.module_44 = py_nndct.nn.ReLU(inplace=True) #SqueezeNet::SqueezeNet/Sequential[features]/Fire[10]/ReLU[expand1x1_activation]/3005
        self.module_45 = py_nndct.nn.Conv2d(in_channels=48, out_channels=192, kernel_size=[3, 3], stride=[1, 1], padding=[1, 1], dilation=[1, 1], groups=1, bias=True) #SqueezeNet::SqueezeNet/Sequential[features]/Fire[10]/Conv2d[expand3x3]/input.63
        self.module_46 = py_nndct.nn.ReLU(inplace=True) #SqueezeNet::SqueezeNet/Sequential[features]/Fire[10]/ReLU[expand3x3_activation]/3025
        self.module_47 = py_nndct.nn.Cat() #SqueezeNet::SqueezeNet/Sequential[features]/Fire[10]/input.65
        self.module_48 = py_nndct.nn.Conv2d(in_channels=384, out_channels=64, kernel_size=[1, 1], stride=[1, 1], padding=[0, 0], dilation=[1, 1], groups=1, bias=True) #SqueezeNet::SqueezeNet/Sequential[features]/Fire[11]/Conv2d[squeeze]/input.67
        self.module_49 = py_nndct.nn.ReLU(inplace=True) #SqueezeNet::SqueezeNet/Sequential[features]/Fire[11]/ReLU[squeeze_activation]/input.69
        self.module_50 = py_nndct.nn.Conv2d(in_channels=64, out_channels=256, kernel_size=[1, 1], stride=[1, 1], padding=[0, 0], dilation=[1, 1], groups=1, bias=True) #SqueezeNet::SqueezeNet/Sequential[features]/Fire[11]/Conv2d[expand1x1]/input.71
        self.module_51 = py_nndct.nn.ReLU(inplace=True) #SqueezeNet::SqueezeNet/Sequential[features]/Fire[11]/ReLU[expand1x1_activation]/3068
        self.module_52 = py_nndct.nn.Conv2d(in_channels=64, out_channels=256, kernel_size=[3, 3], stride=[1, 1], padding=[1, 1], dilation=[1, 1], groups=1, bias=True) #SqueezeNet::SqueezeNet/Sequential[features]/Fire[11]/Conv2d[expand3x3]/input.73
        self.module_53 = py_nndct.nn.ReLU(inplace=True) #SqueezeNet::SqueezeNet/Sequential[features]/Fire[11]/ReLU[expand3x3_activation]/3088
        self.module_54 = py_nndct.nn.Cat() #SqueezeNet::SqueezeNet/Sequential[features]/Fire[11]/input.75
        self.module_55 = py_nndct.nn.Conv2d(in_channels=512, out_channels=64, kernel_size=[1, 1], stride=[1, 1], padding=[0, 0], dilation=[1, 1], groups=1, bias=True) #SqueezeNet::SqueezeNet/Sequential[features]/Fire[12]/Conv2d[squeeze]/input.77
        self.module_56 = py_nndct.nn.ReLU(inplace=True) #SqueezeNet::SqueezeNet/Sequential[features]/Fire[12]/ReLU[squeeze_activation]/input.79
        self.module_57 = py_nndct.nn.Conv2d(in_channels=64, out_channels=256, kernel_size=[1, 1], stride=[1, 1], padding=[0, 0], dilation=[1, 1], groups=1, bias=True) #SqueezeNet::SqueezeNet/Sequential[features]/Fire[12]/Conv2d[expand1x1]/input.81
        self.module_58 = py_nndct.nn.ReLU(inplace=True) #SqueezeNet::SqueezeNet/Sequential[features]/Fire[12]/ReLU[expand1x1_activation]/3131
        self.module_59 = py_nndct.nn.Conv2d(in_channels=64, out_channels=256, kernel_size=[3, 3], stride=[1, 1], padding=[1, 1], dilation=[1, 1], groups=1, bias=True) #SqueezeNet::SqueezeNet/Sequential[features]/Fire[12]/Conv2d[expand3x3]/input.83
        self.module_60 = py_nndct.nn.ReLU(inplace=True) #SqueezeNet::SqueezeNet/Sequential[features]/Fire[12]/ReLU[expand3x3_activation]/3151
        self.module_61 = py_nndct.nn.Cat() #SqueezeNet::SqueezeNet/Sequential[features]/Fire[12]/input.85
        self.module_62 = py_nndct.nn.Conv2d(in_channels=512, out_channels=256, kernel_size=[1, 1], stride=[1, 1], padding=[0, 0], dilation=[1, 1], groups=1, bias=True) #SqueezeNet::SqueezeNet/Sequential[classifier]/Conv2d[0]/input.87
        self.module_63 = py_nndct.nn.ReLU(inplace=True) #SqueezeNet::SqueezeNet/Sequential[classifier]/ReLU[1]/input.89
        self.module_64 = py_nndct.nn.Conv2d(in_channels=256, out_channels=5, kernel_size=[1, 1], stride=[1, 1], padding=[0, 0], dilation=[1, 1], groups=1, bias=True) #SqueezeNet::SqueezeNet/Sequential[classifier]/Conv2d[3]/input
        self.module_65 = py_nndct.nn.AdaptiveAvgPool2d(output_size=1) #SqueezeNet::SqueezeNet/Sequential[classifier]/AdaptiveAvgPool2d[4]/3212
        self.module_66 = py_nndct.nn.Module('flatten') #SqueezeNet::SqueezeNet/3215

    def forward(self, *args):
        output_module_0 = self.module_0(input=args[0])
        output_module_0 = self.module_1(output_module_0)
        output_module_0 = self.module_2(output_module_0)
        output_module_0 = self.module_3(output_module_0)
        output_module_0 = self.module_4(output_module_0)
        output_module_0 = self.module_5(output_module_0)
        output_module_6 = self.module_6(output_module_0)
        output_module_6 = self.module_7(output_module_6)
        output_module_8 = self.module_8(output_module_0)
        output_module_8 = self.module_9(output_module_8)
        output_module_6 = self.module_10(dim=1, tensors=[output_module_6,output_module_8])
        output_module_6 = self.module_11(output_module_6)
        output_module_6 = self.module_12(output_module_6)
        output_module_13 = self.module_13(output_module_6)
        output_module_13 = self.module_14(output_module_13)
        output_module_15 = self.module_15(output_module_6)
        output_module_15 = self.module_16(output_module_15)
        output_module_13 = self.module_17(dim=1, tensors=[output_module_13,output_module_15])
        output_module_13 = self.module_18(output_module_13)
        output_module_13 = self.module_19(output_module_13)
        output_module_13 = self.module_20(output_module_13)
        output_module_21 = self.module_21(output_module_13)
        output_module_21 = self.module_22(output_module_21)
        output_module_23 = self.module_23(output_module_13)
        output_module_23 = self.module_24(output_module_23)
        output_module_21 = self.module_25(dim=1, tensors=[output_module_21,output_module_23])
        output_module_21 = self.module_26(output_module_21)
        output_module_21 = self.module_27(output_module_21)
        output_module_28 = self.module_28(output_module_21)
        output_module_28 = self.module_29(output_module_28)
        output_module_30 = self.module_30(output_module_21)
        output_module_30 = self.module_31(output_module_30)
        output_module_28 = self.module_32(dim=1, tensors=[output_module_28,output_module_30])
        output_module_28 = self.module_33(output_module_28)
        output_module_28 = self.module_34(output_module_28)
        output_module_28 = self.module_35(output_module_28)
        output_module_36 = self.module_36(output_module_28)
        output_module_36 = self.module_37(output_module_36)
        output_module_38 = self.module_38(output_module_28)
        output_module_38 = self.module_39(output_module_38)
        output_module_36 = self.module_40(dim=1, tensors=[output_module_36,output_module_38])
        output_module_36 = self.module_41(output_module_36)
        output_module_36 = self.module_42(output_module_36)
        output_module_43 = self.module_43(output_module_36)
        output_module_43 = self.module_44(output_module_43)
        output_module_45 = self.module_45(output_module_36)
        output_module_45 = self.module_46(output_module_45)
        output_module_43 = self.module_47(dim=1, tensors=[output_module_43,output_module_45])
        output_module_43 = self.module_48(output_module_43)
        output_module_43 = self.module_49(output_module_43)
        output_module_50 = self.module_50(output_module_43)
        output_module_50 = self.module_51(output_module_50)
        output_module_52 = self.module_52(output_module_43)
        output_module_52 = self.module_53(output_module_52)
        output_module_50 = self.module_54(dim=1, tensors=[output_module_50,output_module_52])
        output_module_50 = self.module_55(output_module_50)
        output_module_50 = self.module_56(output_module_50)
        output_module_57 = self.module_57(output_module_50)
        output_module_57 = self.module_58(output_module_57)
        output_module_59 = self.module_59(output_module_50)
        output_module_59 = self.module_60(output_module_59)
        output_module_57 = self.module_61(dim=1, tensors=[output_module_57,output_module_59])
        output_module_57 = self.module_62(output_module_57)
        output_module_57 = self.module_63(output_module_57)
        output_module_57 = self.module_64(output_module_57)
        output_module_57 = self.module_65(output_module_57)
        output_module_57 = self.module_66(input=output_module_57, start_dim=1, end_dim=3)
        return output_module_57
