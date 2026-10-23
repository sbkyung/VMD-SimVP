import torch
from torch import nn


class BasicConv2d(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride, padding, transpose=False, act_norm=False):
        super(BasicConv2d, self).__init__()
        self.act_norm=act_norm
        if not transpose:
            self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, stride=stride, padding=padding, bias=False)
        else:
            self.conv = nn.ConvTranspose2d(in_channels, out_channels, kernel_size=kernel_size, stride=stride, padding=padding,output_padding=stride //2, bias=False )
        self.norm = nn.GroupNorm(2, out_channels)
        self.act = nn.LeakyReLU(0.2, inplace=True)
        self.dropout = nn.Dropout2d(p=0.2)

    def forward(self, x):
        y = self.conv(x)
        if self.act_norm:
            y = self.act(self.norm(y))
        y = self.dropout(y)
        return y

class ConvSC(nn.Module):
    def __init__(self, C_in, C_out, stride, transpose=False, act_norm=True):
        super(ConvSC, self).__init__()
        if stride == 1:
            transpose = False
        self.conv = BasicConv2d(C_in, C_out, kernel_size=3, stride=stride,
                                padding=1, transpose=transpose, act_norm=act_norm)

    def forward(self, x):
        y = self.conv(x)
        return y

class GroupConv2d(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride, padding, groups, act_norm=False):
        super(GroupConv2d, self).__init__()
        self.act_norm = act_norm
        if groups < 1 or in_channels % groups or out_channels % groups:
            raise ValueError("groups must divide both input and output channels")
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, stride=stride, padding=padding,groups=groups, bias=False)
        self.norm = nn.GroupNorm(groups,out_channels)
        self.activate = nn.LeakyReLU(0.2, inplace=True)
        self.dropout = nn.Dropout2d(p=0.2)

    def forward(self, x):
        y = self.conv(x)
        if self.act_norm:
            y = self.activate(self.norm(y))
        y = self.dropout(y)
        return y

class Inception(nn.Module):
    def __init__(self, C_in, C_hid, C_out, incep_ker=[3,5,7,11], groups=8):
        super(Inception, self).__init__()
        self.conv1 = nn.Conv2d(C_in, C_hid, kernel_size=1, stride=1, padding=0, bias=False)
        self.dropout = nn.Dropout2d(p=0.2)

        layers = []
        for ker in incep_ker:
            layers.append(GroupConv2d(C_hid, C_out, kernel_size=ker, stride=1, padding=ker//2, groups=groups, act_norm=True))
        self.layers = nn.Sequential(*layers)

    def forward(self, x):
        x = self.conv1(x)
        x = self.dropout(x)
        y = 0
        for layer in self.layers:
            y += layer(x)
        return y

def stride_generator(N, reverse=False):
    strides = [1, 2]*10
    if reverse: return list(reversed(strides[:N]))
    else: return strides[:N]

class Encoder(nn.Module):
    def __init__(self,C_in, C_hid, N_S):
        super(Encoder,self).__init__()
        strides = stride_generator(N_S)
        self.enc = nn.Sequential(
            ConvSC(C_in, C_hid, stride=strides[0]),
            *[ConvSC(C_hid, C_hid, stride=s) for s in strides[1:]]
        )

    def forward(self,x):
        enc1 = self.enc[0](x)
        latent = enc1
        for i in range(1,len(self.enc)):
            latent = self.enc[i](latent)
        return latent,enc1

class Decoder(nn.Module):
    def __init__(self,C_hid, C_out, N_S, C_hid2=0):
        super(Decoder,self).__init__()
        strides = stride_generator(N_S, reverse=True)

        if N_S>2:
            self.dec = nn.Sequential(
                ConvSC(C_hid+C_hid2, C_hid, stride=strides[0], transpose=True),
                *[ConvSC(C_hid, C_hid, stride=s, transpose=True) for s in strides[1:-1]],
                ConvSC(4*C_hid, C_hid, stride=strides[-1], transpose=True)
            )
        else:
            self.dec = nn.Sequential(
                *[ConvSC(C_hid+C_hid2, C_hid, stride=s, transpose=True) for s in strides[0:-1]],
                ConvSC(4*C_hid, C_hid, stride=strides[-1], transpose=True)
            )
        self.readout = nn.Conv2d(C_hid, C_out, 1)

    def forward(self, hid, enc1=None):
        for i in range(0,len(self.dec)-1):
            hid = self.dec[i](hid)
        Y = self.dec[-1](torch.cat([hid, enc1], dim=1))
        Y = self.readout(Y)
        return Y

class Mid_Xnet(nn.Module):
    def __init__(self, channel_in, channel_hid, N_T, incep_ker = [3,5,7,11], groups=8, channel_in2=0, channel_out=0):
        super(Mid_Xnet, self).__init__()

        if channel_out==0:
            channel_out = channel_in

        self.N_T = N_T
        enc_layers = [Inception(channel_in+channel_in2, channel_hid//2, channel_hid, incep_ker= incep_ker, groups=groups)]
        for i in range(1, N_T-1):
            enc_layers.append(Inception(channel_hid, channel_hid//2, channel_hid, incep_ker= incep_ker, groups=groups))
        enc_layers.append(Inception(channel_hid, channel_hid//2, channel_hid, incep_ker= incep_ker, groups=groups))

        dec_layers = [Inception(channel_hid, channel_hid//2, channel_hid, incep_ker= incep_ker, groups=groups)]
        for i in range(1, N_T-1):
            dec_layers.append(Inception(2*channel_hid, channel_hid//2, channel_hid, incep_ker= incep_ker, groups=groups))
        dec_layers.append(Inception(2*channel_hid, channel_hid//2, channel_out, incep_ker= incep_ker, groups=groups))

        self.enc = nn.Sequential(*enc_layers)
        self.dec = nn.Sequential(*dec_layers)

    def forward(self, x, frct=None):

        B, T, C, H, W = x.shape
        x = x.reshape(B, T*C, H, W)

        skips = []
        z = x

        if frct is not None:
            z = torch.cat([z, frct], dim=1)

        for i in range(self.N_T):
            z = self.enc[i](z)
            if i < self.N_T - 1:
                skips.append(z)

        z = self.dec[0](z)
        for i in range(1, self.N_T):
            z = self.dec[i](torch.cat([z, skips[-i]], dim=1))

        y = z.reshape(B, 12, C, H, W)

        return y

class SimVP(nn.Module):
    def __init__(self, shape_in, hid_S=16, hid_T=256, N_S=4, N_T=8, out_c=3, incep_ker=[3,5,7,11], groups=8):
        super(SimVP, self).__init__()
        T, C, H, W = shape_in

        self.out_c = out_c

        if C==1:
            self.enc = Encoder(1, hid_S, N_S)
            self.hid = Mid_Xnet(T*hid_S, hid_T, N_T, incep_ker, groups, 3*hid_S, 0)

            self.enc_fcst = Encoder(2, hid_S, N_S)
            self.hid_fcst = Mid_Xnet(3*hid_S, hid_T, N_T, incep_ker, groups, 0, T*hid_S)

        else:
            self.enc = Encoder(1, hid_S, N_S)
            self.enc_cond = Encoder(C-1, hid_S, N_S)
            self.hid = Mid_Xnet(T*hid_S, hid_T, N_T, incep_ker, groups, (T+3)*hid_S, 0)
            self.hid_cond = Mid_Xnet(T*hid_S, hid_T, N_T, incep_ker, groups, 0, 0)

            self.enc_fcst = Encoder(2, hid_S, N_S)
            self.hid_fcst = Mid_Xnet(3*hid_S, hid_T, N_T, incep_ker, groups, 0, T*hid_S)

        if C==1:
            self.dec = Decoder(hid_S, 1, N_S, hid_S)
        else:
            self.dec = Decoder(hid_S, 1, N_S, hid_S*2)

    def forward(self, x_raw, forecast):
        B, T, C, H, W = x_raw.shape

        x = x_raw[:,:,0:1,:,:].clone().reshape(B*T, 1, H, W)

        embed, skip = self.enc(x)
        _, C_, H_, W_ = embed.shape

        z = embed.reshape(B, T, C_, H_, W_)

        forecast = forecast.reshape(B*3,2,H,W)
        embed_f, skip_f = self.enc_fcst(forecast)
        z_f = embed_f.reshape(B,3,C_,H_,W_)

        hid_f = self.hid_fcst(z_f)
        z_f = z_f.reshape(B,3*C_,H_,W_)

        hid_f = hid_f.reshape(B*T,C_,H_,W_)

        skip_f = skip_f.reshape(B,3,C_,H,W)
        skip_f = skip_f.repeat_interleave(4, dim=1)
        skip_f = skip_f.reshape(B*T,C_,H,W)

        if C==1:
            hid = self.hid(z,z_f)
            hid = hid.reshape(B*T, C_, H_, W_)

            Y = self.dec(torch.cat([hid,hid_f], axis=1), torch.cat([skip, skip_f], axis=1))
        else:
            x_cond = x_raw[:,:,1:,:,:].clone().reshape(B*T,C-1,H,W)
            embed_c, skip_c = self.enc_cond(x_cond)
            z_c = embed_c.reshape(B,T,C_,H_,W_)
            hid_cond = self.hid_cond(z_c)

            z_c = z_c.reshape(B,T*C_,H_,W_)

            hid = self.hid(z,torch.cat([z_c,z_f], axis=1))

            hid = hid.reshape(B*T, C_, H_, W_)
            hid_cond = hid_cond.reshape(B*T, C_, H_, W_)

            Y = self.dec(torch.cat([hid, hid_cond, hid_f], axis=1), torch.cat([skip,skip_c, skip_f],axis=1))

        Y = Y.reshape(B,T,1,H,W)

        return Y
