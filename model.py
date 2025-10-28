import torch
import torch.nn as nn
import torch.nn.functional as F

class Squeeze_Excitation(nn.Module):
    def __init__(self, channel, r=8):
        super().__init__()

        self.pool = nn.AdaptiveAvgPool2d(1)
        self.net = nn.Sequential(
            nn.Linear(channel, channel // r, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(channel // r, channel, bias=False),
            nn.Sigmoid(),
        )

    def forward(self, inputs):
        b, c, _, _ = inputs.shape
        x = self.pool(inputs).view(b, c)
        x = self.net(x).view(b, c, 1, 1)
        x = inputs * x
        return x

class channel_attention_module(nn.Module):
    def __init__(self, ch, ratio=8):
        super().__init__()

        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)

        self.mlp = nn.Sequential(
            nn.Linear(ch, ch//ratio, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(ch//ratio, ch, bias=False)
        )

        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        x1 = self.avg_pool(x).squeeze(-1).squeeze(-1)
        x1 = self.mlp(x1)

        x2 = self.max_pool(x).squeeze(-1).squeeze(-1)
        x2 = self.mlp(x2)

        feats = x1 + x2
        feats = self.sigmoid(feats).unsqueeze(-1).unsqueeze(-1)
        refined_feats = x * feats

        return refined_feats


class spatial_attention_module(nn.Module):
    def __init__(self, kernel_size=7):
        super().__init__()

        self.conv = nn.Conv2d(2, 1, kernel_size, padding=3, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        x1 = torch.mean(x, dim=1, keepdim=True)
        x2, _ = torch.max(x, dim=1, keepdim=True)

        feats = torch.cat([x1, x2], dim=1)
        feats = self.conv(feats)
        feats = self.sigmoid(feats)
        refined_feats = x * feats

        return refined_feats

class cbam(nn.Module):
    def __init__(self, channel):
        super().__init__()

        self.ca = channel_attention_module(channel)
        self.sa = spatial_attention_module()

    def forward(self, x):
        x = self.ca(x)
        x = self.sa(x)
        return x

class NonLocalBlock(nn.Module):
    def __init__(self, in_channels, inter_channels=None):
        super(NonLocalBlock, self).__init__()
        self.in_channels = in_channels
        self.inter_channels = inter_channels
        if inter_channels is None:
            self.inter_channels = in_channels // 2
            if inter_channels == 0:
                inter_channels = 1
        conv_nd = nn.Conv2d
        max_pool_layer = nn.MaxPool2d(kernel_size=(2, 2))
        bn = nn.BatchNorm2d

        self.g = nn.Conv2d(self.in_channels, self.inter_channels, kernel_size=1)

        self.W_z = conv_nd(in_channels=self.inter_channels, out_channels=self.in_channels, kernel_size=1)

    def forward(self, x):
        batch_size = x.size(0)
        g_x = self.g(x).view(batch_size, self.inter_channels, -1)
        g_x = g_x.permute(0, 2, 1)

        theta_x = x.view(batch_size, self.in_channels, -1)
        phi_x = x.view(batch_size, self.in_channels, -1)
        theta_x = theta_x.permute(0, 2, 1)
        f = torch.matmul(theta_x, phi_x)

        f_div_C = F.softmax(f, dim=-1)

        y = torch.matmul(f_div_C, g_x)

        y = y.permute(0, 2, 1).contiguous()
        y = y.view(batch_size, self.inter_channels, *x.size()[2:])

        W_y = self.W_z(y)
        z = W_y + x
        return z


class GCM(nn.Module):
    def __init__(self, in_c, out_c, rate=[3, 6, 9]):
        super().__init__()

        self.c1 = nn.Sequential(
            nn.Conv2d(in_c, out_c//5, kernel_size=3, dilation=rate[0], padding=rate[0]),
            nn.BatchNorm2d(out_c//5),
            nn.ReLU(inplace=True)
        )

        self.c2 = nn.Sequential(
            nn.Conv2d(in_c, out_c//5, kernel_size=3, dilation=rate[1], padding=rate[1]),
            nn.BatchNorm2d(out_c//5),
            nn.ReLU(inplace=True)
        )

        self.c3 = nn.Sequential(
            nn.Conv2d(in_c, out_c//5, kernel_size=3, dilation=rate[2], padding=rate[2]),
            nn.BatchNorm2d(out_c//5),
            nn.ReLU(inplace=True)
        )

        self.c4 = nn.Sequential(
            nn.Conv2d(in_c, out_c//5, kernel_size=1),
            nn.BatchNorm2d(out_c//5),
            nn.ReLU(inplace=True)

        )



        self.c5 = nn.Conv2d(in_c, out_c//5, kernel_size=1, padding=0)
        self.non_local = NonLocalBlock(out_c//5)
        self.conv1 = nn.Conv2d((out_c//5)*5, out_c, kernel_size=1)

    def forward(self, inputs):
        x1 = self.c1(inputs)
        x2 = self.c2(inputs)
        x3 = self.c3(inputs)
        x4 = self.c4(inputs)
        x5 = self.c5(inputs)
        x5 = self.non_local(x5)
        x = torch.cat((x1, x2, x3, x4, x5), axis=1)
        x = self.conv1(x)
        return x



class MMBA(nn.Module):
    def __init__(self, in_c):
        '''in_channels of encoder part'''
        super().__init__()
        self.conv1 = nn.Conv2d(in_c*2, 1, kernel_size=1)
        self.sigmoid = nn.Sigmoid()
        self.upsample = nn.Upsample(scale_factor=2)
        self.conv_foreground = nn.Conv2d(in_c, in_c//2, kernel_size=3, padding=1)
        self.conv_background = nn.Conv2d(in_c, in_c//2, kernel_size=3, padding=1)
        self.conv_boundary = nn.Conv2d(in_c, in_c//2, kernel_size=3, padding=1)
        self.conv3 = nn.Conv2d(3*(in_c//2), in_c, kernel_size=3, padding=1)
        self.se = Squeeze_Excitation(in_c)

    def forward(self, e, d):
        # e.g. encoder = torch.randn(2, 16, 32, 32) , decoder = torch.randn(2, 32, 16, 16)
        d = self.conv1(d)
        d = self.sigmoid(d)
        d = self.upsample(d)

        foreground = d * e
        background = e * (1-d)
        boundary = e * (1 - (2 * torch.abs(d-0.5)))

        foreground = self.conv_foreground(foreground)
        background = self.conv_background(background)
        boundary = self.conv_boundary(boundary)

        out = torch.cat((foreground, background, boundary), dim=1)   # 3/2
        out = self.conv3(out)
        out = self.se(out)
        out = e + out

        return out



class Afa(nn.Module):
    def __init__(self, up_rate, in_c ):
        # in_c: number of encoder's channels
        super().__init__()
        self.in_c = in_c
        self.upsample1 = nn.Upsample(scale_factor=up_rate)
        self.non_local = NonLocalBlock(in_c)

        self.conv_gcm = nn.Conv2d(1024, in_c, kernel_size=1)

        self.dec_upsample = nn.Upsample(scale_factor=2)
        self.dec_conv = nn.Conv2d(in_c*2, in_c, kernel_size=1)


        self.se = Squeeze_Excitation(in_c)

        self.conv_features = nn.Sequential(
            nn.Conv2d(in_c*3, in_c, kernel_size=3, padding=1),
            nn.BatchNorm2d(in_c),
            nn.ReLU(inplace=True)

        )

        self.conv2 = nn.Sequential(
            nn.Conv2d(in_c, in_c, kernel_size=1),
            nn.BatchNorm2d(in_c),
            nn.ReLU(inplace=True)
        )

    def forward(self, encoder, gcm, decoder):
        gcm = self.upsample1(gcm)
        gcm = self.conv_gcm(gcm)

        decoder = self.dec_upsample(decoder)

        decoder = self.dec_conv(decoder)


        decoder2 = self.non_local(decoder)


        features = torch.cat((encoder, decoder2, gcm), dim=1)
        features = self.conv_features(features)
        features = self.se(features)

        decoder = self.conv2(decoder)

        out = features + decoder
        return out

class conv_block(nn.Module):
    def __init__(self, in_c, out_c):
        super().__init__()

        self.conv1 = nn.Conv2d(in_c, out_c, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(out_c)

        self.conv2 = nn.Conv2d(out_c, out_c, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(out_c)

        self.relu = nn.ReLU()

    def forward(self, inputs):

        x = self.conv1(inputs)
        x = self.bn1(x)
        x = self.relu(x)

        x = self.conv2(x)
        x = self.bn2(x)
        x = self.relu(x)

        return x


class encoder_block(nn.Module):
    def __init__(self, in_c, out_c):
        super().__init__()

        self.conv = conv_block(in_c, out_c)
        self.cbam = cbam(out_c)
        self.pool = nn.MaxPool2d(2)

    def forward(self, inputs):
        x = self.conv(inputs)
        x = self.cbam(x)
        p = self.pool(x)
        return x, p


class deep_supervision(nn.Module):
    def __init__(self, in_c1, in_c2, in_c3, in_c4, in_c5):
        super().__init__()
        self.conv1 = nn.Conv2d(in_c1, 1, kernel_size=1)
        self.upsample1 = nn.Upsample(scale_factor=16)

        self.conv2 = nn.Conv2d(in_c2, 1, kernel_size=1)
        self.upsample2 = nn.Upsample(scale_factor=8)

        self.conv3 = nn.Conv2d(in_c3, 1, kernel_size=1)
        self.upsample3 = nn.Upsample(scale_factor=4)

        self.conv4 = nn.Conv2d(in_c4, 1, kernel_size=1)
        self.upsample4 = nn.Upsample(scale_factor=2)

        self.conv5 = nn.Conv2d(in_c5, 1, kernel_size=1)

        self.conv_out = nn.Conv2d(5, 1, kernel_size=1)



    def forward(self, inp1, inp2, inp3, inp4, inp5):
        out1 = self.conv1(inp1)
        out1 = self.upsample1(out1)

        out2 = self.conv2(inp2)
        out2 = self.upsample2(out2)

        out3 = self.conv3(inp3)
        out3 = self.upsample3(out3)

        out4 = self.conv4(inp4)
        out4 = self.upsample4(out4)

        out5 = self.conv5(inp5)

        out = torch.cat([out1, out2, out3, out4, out5], dim=1)
        out = self.conv_out(out)
        return out




class build_model(nn.Module):
    def __init__(self):
        super().__init__()

        """ Encoder """
        self.e1 = encoder_block(3, 64)
        self.e2 = encoder_block(64, 128)
        self.e3 = encoder_block(128, 256)
        self.e4 = encoder_block(256, 512)
        self.e5 = encoder_block(512, 1024)

        """ Bottleneck """
        self.gcm = GCM(1024, 1024)

        """ Decoder """
        self.d5 = conv_block(1024, 1024)
        self.d4 = conv_block(512, 512)
        self.d3 = conv_block(256, 256)
        self.d2 = conv_block(128, 128)
        self.d1 = conv_block(64, 64)

        self.mmba4 = MMBA(512)
        self.afa4 = Afa(2, 512)

        self.mmba3 = MMBA(256)
        self.afa3 = Afa(4, 256)

        self.mmba2 = MMBA(128)
        self.afa2 = Afa(8, 128)

        self.mmba1 = MMBA(64)
        self.afa1 = Afa(16, 64)

        self.deep_supervision = deep_supervision(1024, 512, 256, 128, 64)

        """ Classifier """

    def forward(self, inputs):
        """ Encoder """
        s1, p1 = self.e1(inputs)    # s1.shape torch.Size([2, 64, 256, 256]), s1.shape torch.Size([2, 64, 128, 128])
        s2, p2 = self.e2(p1)        # s2.shape torch.Size([2, 128, 128, 128])
        s3, p3 = self.e3(p2)        # s3.shape torch.Size([2, 256, 64, 64])
        s4, p4 = self.e4(p3)        # s4.shape torch.Size([2, 512, 32, 32])
        s5, p5 = self.e5(p4)        # s5.shape torch.Size([2, 1024, 16, 16])

        """ Bottleneck """
        gcm = self.gcm(s5)          # gcm.shape:  torch.Size([2, 1024, 16, 16])

        """ Decoder """
        d5 = self.d5(s5)            # d5.shape:  torch.Size([2, 1024, 16, 16])


        mmba4 = self.mmba4(s4, d5)   # mmba4.shape:  torch.Size([2, 512, 32, 32])
        afa4 = self.afa4(mmba4, gcm, d5)    # afa4.shape:  torch.Size([2, 512, 32, 32])

        d4 = self.d4(afa4)         # d4.shape torch.Size([2, 512, 32, 32])

        mmba3 = self.mmba3(s3, d4)        # mmba3.shape:  torch.Size([2, 256, 64, 64])

        afa3 = self.afa3(mmba3, gcm, d4)    # afa3.shape:  torch.Size([2, 256, 64, 64])

        d3 = self.d3(afa3)         # d3.shape:  torch.Size([2, 256, 64, 64])


        mmba2 = self.mmba2(s2, d3)         # mmba.shape:  torch.Size([2, 128, 128, 128])
        afa2 = self.afa2(mmba2, gcm, d3)   # afa2.shape:  torch.Size([2, 128, 128, 128])

        d2 = self.d2(afa2)   # d2.shape:  torch.Size([2, 128, 128, 128])

        mmba1 = self.mmba1(s1, d2)
        afa1 = self.afa1(mmba1, gcm, d2)
        d1 = self.d1(afa1)   # d1.shape:  torch.Size([2, 64, 256, 256])



        # """ Classifier """
        outputs = self.deep_supervision(d5, d4, d3, d2, d1)

        return outputs
