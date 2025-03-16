import torch
import torch.nn.functional as F
from torch import nn
from torchvision import models
from einops import rearrange
from torch.nn.modules.transformer import _get_clones
from mmcv.cnn import ConvModule
from torchvision.models.resnet import resnet101
import timm
from thop import clever_format
from thop import profile

class ASPP(nn.Module): # deeplab

    def __init__(self, dim,in_dim):
        super(ASPP, self).__init__()
        self.down_conv = nn.Sequential(nn.Conv2d(dim,in_dim , 3,padding=1),nn.BatchNorm2d(in_dim),
             nn.ReLU(inplace=False))
        down_dim = in_dim // 2

        self.conv1 = nn.Sequential(
            nn.Conv2d(in_dim, down_dim, kernel_size=1), nn.BatchNorm2d(down_dim), nn.ReLU(inplace=False)
        )
        self.conv2 = nn.Sequential(
            nn.Conv2d(in_dim, down_dim, kernel_size=3, dilation=6, padding=6), nn.BatchNorm2d(down_dim), nn.ReLU(inplace=False)
        )
        self.conv3 = nn.Sequential(
            nn.Conv2d(in_dim, down_dim, kernel_size=3, dilation=12, padding=12), nn.BatchNorm2d(down_dim), nn.ReLU(inplace=False)
        )
        self.conv4 = nn.Sequential(
            nn.Conv2d(in_dim, down_dim, kernel_size=3, dilation=18, padding=18), nn.BatchNorm2d(down_dim), nn.ReLU(inplace=False)
         )
        self.conv5 = nn.Sequential(
            nn.Conv2d(in_dim, down_dim, kernel_size=1),nn.BatchNorm2d(down_dim),  nn.ReLU(inplace=False)
        )
        self.fuse = nn.Sequential(
            nn.Conv2d(5 * down_dim, in_dim, kernel_size=1), nn.BatchNorm2d(in_dim), nn.ReLU(inplace=False)
        )

    def forward(self, x):
        x = self.down_conv(x)
        conv1 = self.conv1(x)
        conv2 = self.conv2(x)
        conv3 = self.conv3(x)
        conv4 = self.conv4(x)
        conv5 = F.upsample(self.conv5(F.adaptive_avg_pool2d(x, 1)), size=x.size()[2:], mode='bilinear')
        return self.fuse(torch.cat((conv1, conv2, conv3,conv4, conv5), 1))

class dynamic_conv(nn.Module):

    def __init__(self,
                 channels,
                 kernel_size,
                 stride):
        super(dynamic_conv, self).__init__()
        self.kernel_size = kernel_size
        self.stride = stride
        self.channels = channels
        reduction_ratio = 4
        self.group_channels = 16
        self.groups = self.channels // self.group_channels
        self.conv1 = ConvModule(
            in_channels=channels,
            out_channels=channels // reduction_ratio,
            kernel_size=1,
            conv_cfg=None,
            norm_cfg=dict(type='BN'),
            act_cfg=dict(type='ReLU'))
        self.conv2 = ConvModule(
            in_channels=channels // reduction_ratio,
            out_channels=kernel_size**2 * self.groups,
            kernel_size=1,
            stride=1,
            conv_cfg=None,
            norm_cfg=None,
            act_cfg=None)
        if stride > 1:
            self.avgpool = nn.AvgPool2d(stride, stride)
        self.unfold = nn.Unfold(kernel_size, 1, (kernel_size-1)//2, stride)

    def forward(self, x, y):
        weight = self.conv2(self.conv1(x if self.stride == 1 else self.avgpool(x)))
        b, c, h, w = weight.shape
        weight = weight.view(b, self.groups, self.kernel_size**2, h, w).unsqueeze(2)
        out = self.unfold(y).view(b, self.groups, self.group_channels, self.kernel_size**2, h, w)
        out = (weight * out).sum(dim=3).view(b, self.channels, h, w)
        return out


class FFN(nn.Module):
    def __init__(self, in_features, hidden_features=None, out_features=None, dropout=0.0):
        """
        two mlp, fc-relu-dropout-fc-relu-drop
        """
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = nn.ReLU(True)
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.dropout(x)
        x = self.fc2(x)
        x = self.dropout(x)
        return x
class TransformerEncoderLayer_modal_filter_v2(nn.Module):
    def __init__(self, d_model, d_ffn, nhead, in_channel=128, dropout=0, drop_path=0, pre_ln=False):
        super().__init__()
        self.norm_attn = nn.LayerNorm(d_model)
        self.self_attn = nn.MultiheadAttention(embed_dim=d_model, num_heads=nhead, dropout=dropout)
        self.norm_ffn = nn.LayerNorm(d_model)
        self.ffn = FFN(in_features=d_model, hidden_features=d_ffn, dropout=dropout)
        self.drop_path = DropPath(drop_path) if drop_path > 0.0 else nn.Identity()
        self.pre_ln = pre_ln
        self.depth_fuse_out_branch = nn.Sequential(nn.Conv2d(in_channel*3, in_channel, kernel_size=3, padding=1), nn.BatchNorm2d(in_channel), nn.ReLU(inplace=False))

        self.contour_fuse_out_branch = nn.Sequential(nn.Conv2d(in_channel*3, in_channel, kernel_size=3, padding=1), nn.BatchNorm2d(in_channel), nn.ReLU(inplace=False))

        self.sal_fuse_out_branch = nn.Sequential(nn.Conv2d(in_channel*3, in_channel, kernel_size=3, padding=1), nn.BatchNorm2d(in_channel), nn.ReLU(inplace=False))

        self.depth_fuse_out_filter_3 = dynamic_conv(in_channel, 3, 1)

        self.contour_fuse_out_filter_3 = dynamic_conv(in_channel, 3, 1)

        self.sal_fuse_out_filter_3 = dynamic_conv(in_channel, 3, 1)

        self.modal_filter_fuse = nn.Sequential(nn.Conv2d(in_channel*3, in_channel*3, kernel_size=3, padding=1), nn.BatchNorm2d(in_channel*3), nn.ReLU(inplace=False))
    def forward(self, src,rgb_w,depth,sal,contour,in_channel):
        if self.pre_ln:
            pre = self.norm_attn(src)
            src = src + self.drop_path(self.self_attn(query=pre, key=pre, value=pre)[0])
            pre = self.norm_ffn(src)
            src = src + self.drop_path(self.ffn(pre))
        else:
            src = self.norm_attn(src + self.drop_path(self.self_attn(query=src, key=src, value=src)[0]))
            src = self.norm_ffn(src + self.drop_path(self.ffn(src)))
            # print(src.shape,rgb_w)
            multi_modal_out = rearrange(src, "(h w) bs c -> bs c h w", w=rgb_w)
            # print(multi_modal_out.shape)
            depth_fuse_out = self.depth_fuse_out_branch(multi_modal_out)
            contour_fuse_out = self.contour_fuse_out_branch(multi_modal_out)
            sal_fuse_out = self.sal_fuse_out_branch(multi_modal_out)
            depth_fuse_out_filter_3 = self.depth_fuse_out_filter_3(depth, depth_fuse_out)


            contour_fuse_out_filter_3 = self.contour_fuse_out_filter_3(contour, contour_fuse_out)
            sal_fuse_out_filter_3 = self.sal_fuse_out_filter_3(sal, sal_fuse_out)
            modal_filter_fuse = self.modal_filter_fuse(torch.cat((depth_fuse_out_filter_3,contour_fuse_out_filter_3,sal_fuse_out_filter_3),1))

        return depth_fuse_out_filter_3, contour_fuse_out_filter_3, sal_fuse_out_filter_3, modal_filter_fuse


class Transformer_with_modal_filter(nn.Module):
    def __init__(self, num_enc_layers, num_dec_layers, nhead, d_model, d_ffn, pre_ln, area, dropout=0, drop_path=0):
        super().__init__()
        assert not pre_ln
        encoder_layer = TransformerEncoderLayer_modal_filter_v2(
            nhead=nhead, d_model=d_model, d_ffn=d_ffn, dropout=dropout, drop_path=drop_path, pre_ln=pre_ln
        )
        self.rgb_encoder_layers = _get_clones(encoder_layer, num_enc_layers)


    def forward(self, rgb_feats, depth, sal, contour):
        """
        Args:
            rgb_feats: N,C,H,W
            depth_feats: N,C,H,W
        Returns:
            N,C,H,W
        """
        rgb_w = rgb_feats.shape[-1]
        rgb_feats = rearrange(rgb_feats, "bs c h w -> (h w) bs c")
        rgb = rgb_feats
        rgb_outs = []
        depth_outs = []
        contour_outs = []
        sal_outs = []
        for i, rgb_layer in enumerate(self.rgb_encoder_layers):
            depth_f, contour_f, sal_f, rgb_reshape = rgb_layer(rgb,rgb_w,depth,sal,contour,128)
            rgb = rearrange(rgb_reshape, "bs c h w -> (h w) bs c")
            rgb_outs.append(rgb_reshape)
            depth_outs.append(depth_f)
            contour_outs.append(contour_f)
            sal_outs.append(sal_f)

        rgb_out = torch.cat(rgb_outs, dim=1)
        depth_out = torch.cat(depth_outs, dim=1)
        contour_out = torch.cat(contour_outs, dim=1)
        sal_out = torch.cat(sal_outs, dim=1)
        return depth_out, contour_out, sal_out


class baseline(nn.Module):

    def __init__(self):
        super(baseline, self).__init__()
        ################################vgg16#######################################
        feats = list(models.vgg16_bn(pretrained=True).features.children())
        self.conv1_RGB = nn.Sequential(*feats[0:6])
        self.conv2_RGB = nn.Sequential(*feats[6:13])
        self.conv3_RGB = nn.Sequential(*feats[13:23])
        self.conv4_RGB = nn.Sequential(*feats[23:33])
        self.conv5_RGB = nn.Sequential(*feats[33:43])

        self.dem3 = nn.Sequential(nn.Conv2d(256, 128, kernel_size=3, padding=1), nn.BatchNorm2d(128),
                                  nn.ReLU(inplace=False))
        self.dem4 = nn.Sequential(nn.Conv2d(512, 128, kernel_size=3, padding=1), nn.BatchNorm2d(128),
                                  nn.ReLU(inplace=False))
        self.dem5 = nn.Sequential(nn.Conv2d(512, 128, kernel_size=3, padding=1), nn.BatchNorm2d(128),
                                  nn.ReLU(inplace=False))

        self.output4_sal = nn.Sequential(nn.Conv2d(128, 128, kernel_size=3, padding=1), nn.BatchNorm2d(128),
                                         nn.ReLU(inplace=False))
        self.output3_sal = nn.Sequential(nn.Conv2d(128, 128, kernel_size=3, padding=1), nn.BatchNorm2d(128),
                                         nn.ReLU(inplace=False))
        self.output2_sal = nn.Sequential(nn.Conv2d(128, 128, kernel_size=3, padding=1), nn.BatchNorm2d(128),
                                         nn.ReLU(inplace=False))
        self.output1_sal = nn.Sequential(nn.Conv2d(64, 64, kernel_size=3, padding=1), nn.BatchNorm2d(64),
                                         nn.ReLU(inplace=False))

        for m in self.modules():
            if isinstance(m, nn.ReLU) or isinstance(m, nn.Dropout):
                m.inplace = True

    def forward(self, x):
        input = x
        B, _, _, _ = input.size()
        e1_rgb = self.conv1_RGB(x)
        e2_rgb = self.conv2_RGB(e1_rgb)
        e3_rgb = self.conv3_RGB(e2_rgb)
        e4_rgb = self.conv4_RGB(e3_rgb)
        e5_rgb = self.conv5_RGB(e4_rgb)

        e3_rgb = self.dem3(e3_rgb)
        e4_rgb = self.dem4(e4_rgb)
        e5_rgb = self.dem5(e5_rgb)
        output4_sal = self.output4_sal(F.upsample(e5_rgb, size=e4_rgb.size()[2:], mode='bilinear') + e4_rgb)
        output3_sal = self.output3_sal(F.upsample(output4_sal, size=e3_rgb.size()[2:], mode='bilinear') + e3_rgb)
        output2_sal = self.output2_sal(F.upsample(output3_sal, size=e2_rgb.size()[2:], mode='bilinear') + e2_rgb)
        output1_sal = self.output1_sal(F.upsample(output2_sal, size=e1_rgb.size()[2:], mode='bilinear') + e1_rgb)
        return output1_sal


class MMFT(nn.Module):

    def __init__(self):
        super(MMFT, self).__init__()
        ################################vgg16#######################################
        # self.bkbone = timm.create_model('res2net50_26w_8s', features_only=True, pretrained=True)
        # self.bkbone = timm.create_model('resnet101d', features_only=True, pretrained=True)
        self.bkbone = timm.create_model('resnet50', features_only=True, pretrained=True)

        self.dem2 = nn.Sequential(nn.Conv2d(256, 128, kernel_size=3, padding=1), nn.BatchNorm2d(128),
                                  nn.ReLU(inplace=False))
        self.dem3 = nn.Sequential(nn.Conv2d(512, 128, kernel_size=3, padding=1), nn.BatchNorm2d(128),
                                  nn.ReLU(inplace=False))
        self.dem4 = nn.Sequential(nn.Conv2d(1024, 128, kernel_size=3, padding=1), nn.BatchNorm2d(128),
                                  nn.ReLU(inplace=False))
        self.dem5 = nn.Sequential(nn.Conv2d(2048, 128, kernel_size=3, padding=1), nn.BatchNorm2d(128),
                                  nn.ReLU(inplace=False))
        self.ASPP = ASPP(2048,128)
        self.Ini_Depth_F = nn.Sequential(nn.Conv2d(128+64, 128, kernel_size=3, padding=1), nn.BatchNorm2d(128), nn.ReLU(inplace=False))
        self.Ini_Contour_F = nn.Sequential(nn.Conv2d(128+64, 128, kernel_size=3, padding=1), nn.BatchNorm2d(128), nn.ReLU(inplace=False))
        self.Ini_Sal_F = nn.Sequential(nn.Conv2d(128+64, 128, kernel_size=3, padding=1), nn.BatchNorm2d(128), nn.ReLU(inplace=False))

        self.sideout_Depth_5 = nn.Conv2d(128, 1, kernel_size=3, padding=1)
        self.sideout_Contour_5 = nn.Conv2d(128, 1, kernel_size=3, padding=1)
        self.sideout_Sal_5 = nn.Conv2d(128, 1, kernel_size=3, padding=1)
        self.sideout_Depth_4 = nn.Conv2d(128, 1, kernel_size=3, padding=1)
        self.sideout_Contour_4 = nn.Conv2d(128, 1, kernel_size=3, padding=1)
        self.sideout_Sal_4 = nn.Conv2d(128, 1, kernel_size=3, padding=1)
        self.sideout_Depth_3 = nn.Conv2d(128, 1, kernel_size=3, padding=1)
        self.sideout_Contour_3 = nn.Conv2d(128, 1, kernel_size=3, padding=1)
        self.sideout_Sal_3 = nn.Conv2d(128, 1, kernel_size=3, padding=1)
        self.sideout_Depth_2 = nn.Conv2d(128, 1, kernel_size=3, padding=1)
        self.sideout_Contour_2 = nn.Conv2d(128, 1, kernel_size=3, padding=1)
        self.sideout_Sal_2 = nn.Conv2d(128, 1, kernel_size=3, padding=1)
        self.sideout_Depth_1 = nn.Conv2d(64, 1, kernel_size=3, padding=1)
        self.sideout_Contour_1 = nn.Conv2d(64, 1, kernel_size=3, padding=1)
        self.sideout_Sal_1 = nn.Conv2d(64, 1, kernel_size=3, padding=1)

        self.multi_modal_fuse_depth_5 = nn.Sequential(nn.Conv2d(768, 128, kernel_size=3, padding=1), nn.BatchNorm2d(128), nn.ReLU(inplace=False))
        self.multi_modal_fuse_sal_5 = nn.Sequential(nn.Conv2d(768, 128, kernel_size=3, padding=1), nn.BatchNorm2d(128), nn.ReLU(inplace=False))
        self.multi_modal_fuse_contour_5 = nn.Sequential(nn.Conv2d(768, 128, kernel_size=3, padding=1), nn.BatchNorm2d(128), nn.ReLU(inplace=False))
        self.depth5_res = nn.Sequential(nn.Conv2d(128, 128, kernel_size=3, padding=1), nn.BatchNorm2d(128), nn.ReLU(inplace=False))
        self.contour5_res = nn.Sequential(nn.Conv2d(128, 128, kernel_size=3, padding=1), nn.BatchNorm2d(128), nn.ReLU(inplace=False))
        self.sal5_res = nn.Sequential(nn.Conv2d(128, 128, kernel_size=3, padding=1), nn.BatchNorm2d(128), nn.ReLU(inplace=False))

        self.multi_modal_fuse_4 = nn.Sequential(nn.Conv2d(128, 128, kernel_size=3, padding=1), nn.BatchNorm2d(128), nn.ReLU(inplace=False))
        self.multi_modal_fuse_depth_4 = nn.Sequential(nn.Conv2d(128, 128, kernel_size=3, padding=1), nn.BatchNorm2d(128), nn.ReLU(inplace=False))
        self.multi_modal_fuse_sal_4 = nn.Sequential(nn.Conv2d(128, 128, kernel_size=3, padding=1), nn.BatchNorm2d(128), nn.ReLU(inplace=False))
        self.multi_modal_fuse_contour_4 = nn.Sequential(nn.Conv2d(128, 128, kernel_size=3, padding=1), nn.BatchNorm2d(128), nn.ReLU(inplace=False))
        self.depth4_res = nn.Sequential(nn.Conv2d(128, 128, kernel_size=3, padding=1), nn.BatchNorm2d(128), nn.ReLU(inplace=False))
        self.contour4_res = nn.Sequential(nn.Conv2d(128, 128, kernel_size=3, padding=1), nn.BatchNorm2d(128), nn.ReLU(inplace=False))
        self.sal4_res = nn.Sequential(nn.Conv2d(128, 128, kernel_size=3, padding=1), nn.BatchNorm2d(128), nn.ReLU(inplace=False))

        self.multi_modal_fuse_3 = nn.Sequential(nn.Conv2d(128, 128, kernel_size=3, padding=1), nn.BatchNorm2d(128), nn.ReLU(inplace=False))
        self.multi_modal_fuse_depth_3 = nn.Sequential(nn.Conv2d(128, 128, kernel_size=3, padding=1), nn.BatchNorm2d(128), nn.ReLU(inplace=False))
        self.multi_modal_fuse_sal_3 = nn.Sequential(nn.Conv2d(128, 128, kernel_size=3, padding=1), nn.BatchNorm2d(128), nn.ReLU(inplace=False))
        self.multi_modal_fuse_contour_3 = nn.Sequential(nn.Conv2d(128, 128, kernel_size=3, padding=1), nn.BatchNorm2d(128), nn.ReLU(inplace=False))
        self.depth3_res = nn.Sequential(nn.Conv2d(128, 128, kernel_size=3, padding=1), nn.BatchNorm2d(128), nn.ReLU(inplace=False))
        self.contour3_res = nn.Sequential(nn.Conv2d(128, 128, kernel_size=3, padding=1), nn.BatchNorm2d(128), nn.ReLU(inplace=False))
        self.sal3_res = nn.Sequential(nn.Conv2d(128, 128, kernel_size=3, padding=1), nn.BatchNorm2d(128), nn.ReLU(inplace=False))

        self.multi_modal_fuse_2 = nn.Sequential(nn.Conv2d(128, 128, kernel_size=3, padding=1), nn.BatchNorm2d(128), nn.ReLU(inplace=False))
        self.multi_modal_fuse_depth_2 = nn.Sequential(nn.Conv2d(128, 128, kernel_size=3, padding=1), nn.BatchNorm2d(128), nn.ReLU(inplace=False))
        self.multi_modal_fuse_sal_2 = nn.Sequential(nn.Conv2d(128, 128, kernel_size=3, padding=1), nn.BatchNorm2d(128), nn.ReLU(inplace=False))
        self.multi_modal_fuse_contour_2 = nn.Sequential(nn.Conv2d(128, 128, kernel_size=3, padding=1), nn.BatchNorm2d(128), nn.ReLU(inplace=False))
        self.depth2_res = nn.Sequential(nn.Conv2d(128, 64, kernel_size=3, padding=1), nn.BatchNorm2d(64), nn.ReLU(inplace=False))
        self.contour2_res = nn.Sequential(nn.Conv2d(128, 64, kernel_size=3, padding=1), nn.BatchNorm2d(64), nn.ReLU(inplace=False))
        self.sal2_res = nn.Sequential(nn.Conv2d(128, 64, kernel_size=3, padding=1), nn.BatchNorm2d(64), nn.ReLU(inplace=False))

        self.multi_modal_fuse_1 = nn.Sequential(nn.Conv2d(64, 64, kernel_size=3, padding=1), nn.BatchNorm2d(64), nn.ReLU(inplace=False))
        self.multi_modal_fuse_depth_1 = nn.Sequential(nn.Conv2d(64, 64, kernel_size=3, padding=1), nn.BatchNorm2d(64), nn.ReLU(inplace=False))
        self.multi_modal_fuse_sal_1 = nn.Sequential(nn.Conv2d(64, 64, kernel_size=3, padding=1), nn.BatchNorm2d(64), nn.ReLU(inplace=False))
        self.multi_modal_fuse_contour_1 = nn.Sequential(nn.Conv2d(64, 64, kernel_size=3, padding=1), nn.BatchNorm2d(64), nn.ReLU(inplace=False))
        self.depth1_res =  nn.Conv2d(64, 1, kernel_size=3, padding=1)
        self.contour1_res =  nn.Conv2d(64, 1, kernel_size=3, padding=1)
        self.sal1_res =  nn.Conv2d(64, 1, kernel_size=3, padding=1)


        self.output4_depth = nn.Sequential(nn.Conv2d(128, 128, kernel_size=3, padding=1), nn.BatchNorm2d(128), nn.ReLU(inplace=False))
        self.output3_depth = nn.Sequential(nn.Conv2d(128, 128, kernel_size=3, padding=1), nn.BatchNorm2d(128), nn.ReLU(inplace=False))
        self.output2_depth = nn.Sequential(nn.Conv2d(128, 128, kernel_size=3, padding=1),nn.BatchNorm2d(128), nn.ReLU(inplace=False))
        self.output1_depth = nn.Sequential(nn.Conv2d(64, 64, kernel_size=3, padding=1),nn.BatchNorm2d(64), nn.ReLU(inplace=False))

        self.output4_contour = nn.Sequential(nn.Conv2d(128, 128, kernel_size=3, padding=1), nn.BatchNorm2d(128),
                                           nn.ReLU(inplace=False))
        self.output3_contour = nn.Sequential(nn.Conv2d(128, 128, kernel_size=3, padding=1), nn.BatchNorm2d(128),
                                           nn.ReLU(inplace=False))
        self.output2_contour = nn.Sequential(nn.Conv2d(128, 128, kernel_size=3, padding=1), nn.BatchNorm2d(128),
                                           nn.ReLU(inplace=False))
        self.output1_contour = nn.Sequential(nn.Conv2d(64, 64, kernel_size=3, padding=1),nn.BatchNorm2d(64), nn.ReLU(inplace=False))


        self.output4_sal = nn.Sequential(nn.Conv2d(128, 128, kernel_size=3, padding=1), nn.BatchNorm2d(128), nn.ReLU(inplace=False))
        self.output3_sal = nn.Sequential(nn.Conv2d(128, 128, kernel_size=3, padding=1), nn.BatchNorm2d(128), nn.ReLU(inplace=False))
        self.output2_sal = nn.Sequential(nn.Conv2d(128, 128, kernel_size=3, padding=1),nn.BatchNorm2d(128), nn.ReLU(inplace=False))
        self.output1_sal = nn.Sequential(nn.Conv2d(64, 64, kernel_size=3, padding=1),nn.BatchNorm2d(64), nn.ReLU(inplace=False))


        self.transformer = Transformer_with_modal_filter(
            num_enc_layers=6,
            num_dec_layers=6,
            nhead=4,
            d_model=128*3,
            d_ffn=2 * 64,
            pre_ln=False,
            area=(256 // 16) ** 2 * 3,
        )

        for m in self.modules():
            if isinstance(m, nn.ReLU) or isinstance(m, nn.Dropout):
                m.inplace = True

    def forward(self, x):
        input = x
        B,_,_,_ = input.size()
        e1_rgb, e2_rgb, e3_rgb, e4_rgb, e5_rgb = self.bkbone(x)

        e2_rgb = self.dem2(e2_rgb)
        e3_rgb = self.dem3(e3_rgb)
        e4_rgb = self.dem4(e4_rgb)
        e5_rgb = self.ASPP(e5_rgb)

        Ini_Depth_F = self.Ini_Depth_F(torch.cat([e5_rgb,F.upsample(e1_rgb, size=e5_rgb.size()[2:])],1))
        Ini_Contour_F = self.Ini_Contour_F(torch.cat([e5_rgb,F.upsample(e1_rgb, size=e5_rgb.size()[2:])],1))
        Ini_Sal_F = self.Ini_Sal_F(torch.cat([e5_rgb,F.upsample(e1_rgb, size=e5_rgb.size()[2:])],1))

        sideout_Depth_5 = self.sideout_Depth_5(Ini_Depth_F)
        sideout_Contour_5 = self.sideout_Contour_5(Ini_Contour_F)
        sideout_Sal_5 = self.sideout_Sal_5(Ini_Sal_F)

        transformer_5_input = torch.cat((Ini_Depth_F, Ini_Contour_F, Ini_Sal_F), 1)
        transformer_5_output_depth, transformer_5_output_contour, transformer_5_output_sal = self.transformer(transformer_5_input,Ini_Depth_F,Ini_Sal_F,Ini_Contour_F)
        multi_modal_fuse_depth_5 =  self.multi_modal_fuse_depth_5(transformer_5_output_depth)
        multi_modal_fuse_sal_5 = self.multi_modal_fuse_sal_5(transformer_5_output_contour)
        multi_modal_fuse_contour_5 = self.multi_modal_fuse_contour_5(transformer_5_output_sal)
        output4_depth = self.output4_depth(F.upsample(self.depth5_res(multi_modal_fuse_depth_5+Ini_Depth_F), size=e4_rgb.size()[2:], mode='bilinear')+e4_rgb)
        output4_contour = self.output4_contour(
            F.upsample(self.contour5_res(multi_modal_fuse_contour_5+Ini_Contour_F), size=e4_rgb.size()[2:], mode='bilinear') + e4_rgb)
        output4_sal = self.output4_sal(
            F.upsample(self.sal5_res(multi_modal_fuse_sal_5+Ini_Sal_F), size=e4_rgb.size()[2:], mode='bilinear') + e4_rgb)
        sideout_Depth_4 = self.sideout_Depth_4(output4_depth)
        sideout_Contour_4 = self.sideout_Contour_4(output4_contour)
        sideout_Sal_4 = self.sideout_Sal_4(output4_sal)
        multi_modal_fuse_4 = self.multi_modal_fuse_4(output4_depth+output4_contour+output4_sal)
        multi_modal_fuse_depth_4 = self.multi_modal_fuse_depth_4(multi_modal_fuse_4)
        multi_modal_fuse_sal_4 = self.multi_modal_fuse_sal_4(multi_modal_fuse_4)
        multi_modal_fuse_contour_4 = self.multi_modal_fuse_contour_4(multi_modal_fuse_4)
        output3_depth = self.output3_depth(F.upsample(self.depth4_res(multi_modal_fuse_depth_4+output4_depth), size=e3_rgb.size()[2:], mode='bilinear')+e3_rgb)
        output3_contour = self.output3_contour(
            F.upsample(self.contour4_res(multi_modal_fuse_contour_4+output4_contour), size=e3_rgb.size()[2:], mode='bilinear') + e3_rgb)
        output3_sal = self.output3_sal(F.upsample(self.sal4_res(multi_modal_fuse_sal_4+output4_sal), size=e3_rgb.size()[2:], mode='bilinear') + e3_rgb)
        sideout_Depth_3 = self.sideout_Depth_3(output3_depth)
        sideout_Contour_3 = self.sideout_Contour_3(output3_contour)
        sideout_Sal_3 = self.sideout_Sal_3(output3_sal)
        multi_modal_fuse_3 = self.multi_modal_fuse_3(output3_depth+output3_contour+output3_sal)
        multi_modal_fuse_depth_3 = self.multi_modal_fuse_depth_3(multi_modal_fuse_3)
        multi_modal_fuse_sal_3 = self.multi_modal_fuse_sal_3(multi_modal_fuse_3)
        multi_modal_fuse_contour_3 = self.multi_modal_fuse_contour_3(multi_modal_fuse_3)
        output2_depth = self.output2_depth(F.upsample(self.depth3_res(multi_modal_fuse_depth_3+output3_depth), size=e2_rgb.size()[2:], mode='bilinear') + e2_rgb)
        output2_contour = self.output2_contour(
            F.upsample(self.contour3_res(multi_modal_fuse_contour_3+output3_contour), size=e2_rgb.size()[2:], mode='bilinear') + e2_rgb)
        output2_sal = self.output2_sal(F.upsample(self.sal3_res(multi_modal_fuse_sal_3+output3_sal), size=e2_rgb.size()[2:], mode='bilinear') + e2_rgb)
        sideout_Depth_2 = self.sideout_Depth_2(output2_depth)
        sideout_Contour_2 = self.sideout_Contour_2(output2_contour)
        sideout_Sal_2 = self.sideout_Sal_2(output2_sal)
        multi_modal_fuse_2 = self.multi_modal_fuse_2(output2_depth+output2_contour+output2_sal)
        multi_modal_fuse_depth_2 = self.multi_modal_fuse_depth_2(multi_modal_fuse_2)
        multi_modal_fuse_sal_2 = self.multi_modal_fuse_sal_2(multi_modal_fuse_2)
        multi_modal_fuse_contour_2 = self.multi_modal_fuse_contour_2(multi_modal_fuse_2)
        output1_depth = self.output1_depth(F.upsample(self.depth2_res(multi_modal_fuse_depth_2+output2_depth), size=e1_rgb.size()[2:], mode='bilinear')+e1_rgb)
        output1_contour = self.output1_contour(F.upsample(self.contour2_res(multi_modal_fuse_contour_2+output2_contour), size=e1_rgb.size()[2:], mode='bilinear') + e1_rgb)
        output1_sal = self.output1_sal(F.upsample(self.sal2_res(multi_modal_fuse_sal_2+output2_sal), size=e1_rgb.size()[2:], mode='bilinear') + e1_rgb)
        sideout_Depth_1 = self.sideout_Depth_1(output1_depth)
        sideout_Contour_1 = self.sideout_Contour_1(output1_contour)
        sideout_Sal_1 = self.sideout_Sal_1(output1_sal)
        multi_modal_fuse_1 = self.multi_modal_fuse_1(output1_depth+output1_contour+output1_sal)
        multi_modal_fuse_depth_1 = self.multi_modal_fuse_depth_1(multi_modal_fuse_1)
        multi_modal_fuse_sal_1 = self.multi_modal_fuse_sal_1(multi_modal_fuse_1)
        multi_modal_fuse_contour_1 = self.multi_modal_fuse_contour_1(multi_modal_fuse_1)
        output_depth_final = self.depth1_res(multi_modal_fuse_depth_1 + output1_depth)
        output_contour_final = self.contour1_res(multi_modal_fuse_contour_1 + output1_contour)
        output_sal_final = self.sal1_res(multi_modal_fuse_sal_1 + output1_sal)
        if self.training:
            return sideout_Depth_5, sideout_Contour_5, sideout_Sal_5, sideout_Depth_4, sideout_Contour_4, sideout_Sal_4,sideout_Depth_3, sideout_Contour_3, sideout_Sal_3,sideout_Depth_2, sideout_Contour_2, sideout_Sal_2, sideout_Depth_1,sideout_Contour_1,sideout_Sal_1,output_depth_final,output_contour_final,output_sal_final
        # return F.sigmoid(output_depth_final)
        return output_sal_final



if __name__ == "__main__":
    model = MMFT().eval()
    input = torch.randn(1, 3, 256, 256)
    flops, params = profile(model,inputs=(input,))
    flops, params = clever_format([flops, params], "%.3f")
    print(flops,params)