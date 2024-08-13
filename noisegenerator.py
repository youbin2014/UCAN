import functools
import torch
import torch.nn as nn

def make_layer(block, n_layers):
    layers = []
    for _ in range(n_layers):
        layers.append(block())
    return nn.Sequential(*layers)

class channel_attention(nn.Module):
    def __init__(self, in_nc,mid_nc,size):
        super(channel_attention, self).__init__()
        self.pooling=nn.AvgPool2d(size)
        self.linear1=nn.Linear(in_nc,mid_nc)
        self.lrelu = nn.LeakyReLU(negative_slope=0.2, inplace=True)
        self.linear2=nn.Linear(mid_nc,in_nc)
        self.sigmoid=nn.Sigmoid()
    def forward(self,feature):
        b,c,_,_=feature.size()
        att=self.pooling(feature).view(b,c)
        att=self.lrelu(self.linear1(att))
        att=self.linear2(att)
        att=self.sigmoid(att)
        att=att.view(b,c,1,1)
        return feature*att

class Fusion_Block(nn.Module):
    def __init__(self, nf=64, gc=32, bias=True,in_nc=3,size=32):
        super(Fusion_Block, self).__init__()
        # gc: growth channel, i.e. intermediate channels
        self.conv1 = nn.Conv2d(nf, gc, 3, 1, 1, bias=bias)
        self.conv2 = nn.Conv2d(nf + gc, gc, 3, 1, 1, bias=bias)
        self.conv3 = nn.Conv2d(nf + 2 * gc, nf, 3, 1, 1, bias=bias)
        self.conv4 = nn.Conv2d(nf + 3 * gc, gc, 3, 1, 1, bias=bias)
        self.conv_mean = nn.Conv2d(nf + 4 * gc, nf, 3, 1, 1, bias=bias)
        self.conv_variance = nn.Conv2d(nf + 4 * gc, nf, 3, 1, 1, bias=bias)
        self.lrelu = nn.LeakyReLU(negative_slope=0.2, inplace=True)

    def forward(self, x):
        x1 = self.lrelu(self.conv1(x))
        x2 = self.lrelu(self.conv2(torch.cat((x, x1), 1)))
        x3 = self.lrelu(self.conv3(torch.cat((x, x1, x2), 1)))
        x4 = self.lrelu(self.conv4(torch.cat((x, x1, x2, x3), 1)))
        x5 = self.conv_mean(torch.cat((x, x1, x2, x3, x4), 1))
        # variance = self.conv_variance(torch.cat((x, x1, x2, x3, x4), 1))
        return x5

class NoiseGenerator(nn.Module):
    def __init__(self, in_nc, out_nc,BlockNum,nf=16, gc=16,size=32):
        super(NoiseGenerator, self).__init__()
        self.Image_first = nn.Conv2d(in_nc, nf, 3, 1, 1, bias=True)
        self.TRN = Fusion_Block(nf, gc, in_nc=in_nc, size=size)
        self.trunk_conv_mean = nn.Conv2d(nf, out_nc, 3, 1, 1, bias=True)
        self.trunk_conv_variance = nn.Conv2d(nf, out_nc, 3, 1, 1, bias=True)
        self.relu=nn.ReLU()
        self.tanh=nn.Tanh()
        self.BlockNum = BlockNum

    def forward(self, X_adv):
        Image_fea = self.Image_first(X_adv)
        # for i in range(self.BlockNum):
        #     Image_fea = self.TRN[i](Image_fea)
        x = self.TRN(Image_fea)
        x = self.tanh(self.trunk_conv_mean(x))
        return x

class Generator(nn.Module):
    def __init__(self,input_size):
        super(Generator, self).__init__()

        def block(in_feat, out_feat, normalize=False):
            layers = [nn.Linear(in_feat, out_feat)]
            if normalize:
                layers.append(nn.BatchNorm1d(out_feat, 0.8))
            layers.append(nn.LeakyReLU(0.2, inplace=True))
            return layers

        self.model = nn.Sequential(
            *block(1, 128, normalize=False),
            *block(128, 256),
            *block(256, 512),
            *block(512, 1024),
            nn.Linear(1024, int(input_size)),
            nn.Tanh()
        )

    def forward(self, z):
        img = self.model(z)
        img = img.view(img.size(0),-1)
        return img