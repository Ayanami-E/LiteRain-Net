import torch
import torch.nn as nn
import torch.nn.functional as F

# Layer Norm
class LayerNorm(nn.Module):
    def __init__(self, normalized_shape, eps=1e-6, data_format="channels_first"):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(normalized_shape))
        self.bias = nn.Parameter(torch.zeros(normalized_shape))
        self.eps = eps
        self.data_format = data_format
        if self.data_format not in ["channels_last", "channels_first"]:
            raise NotImplementedError
        self.normalized_shape = (normalized_shape, )

    def forward(self, x):
        if self.data_format == "channels_last":
            return F.layer_norm(x, self.normalized_shape, self.weight, self.bias, self.eps)
        elif self.data_format == "channels_first":
            u = x.mean(1, keepdim=True)
            s = (x - u).pow(2).mean(1, keepdim=True)
            x = (x - u) / torch.sqrt(s + self.eps)
            x = self.weight[:, None, None] * x + self.bias[:, None, None]
            return x

# FC
class FC(nn.Module):
    def __init__(self, dim, growth_rate=2.0):
        super().__init__()
        hidden_dim = int(dim * growth_rate)

        self.fc = nn.Sequential(
            nn.Conv2d(dim, hidden_dim, 3, 1, 1),
            nn.GELU(), 
            nn.Conv2d(hidden_dim, dim, 1, 1, 0)
        )

    def forward(self, x):
        return self.fc(x)

# Gobal feature
class Gobal(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.conv1 = nn.Conv2d(dim, dim, 1, 1, 0)
        self.act1 = nn.GELU()
        self.conv2 = nn.Conv2d(dim, dim, 1, 1, 0)
        self.act2 = nn.GELU()
        self.conv3 = nn.Conv2d(dim, dim, 1, 1, 0)
        self.act3 = nn.Sigmoid()

    def forward(self, x):
        _, C, H, W = x.shape
        # print(x.shape,"a")
        y = F.interpolate(x, size=[C, C], mode='bilinear', align_corners=True)
        # print(y.shape, "b")
        # b c h w -> b h w c
        y = self.act1(self.conv1(y)).permute(0, 2, 3, 1)
        # b h w c -> b w c h
        y = self.act2(self.conv2(y)).permute(0, 2, 3, 1)
        # b w c h -> b c h w
        y = self.act3(self.conv3(y)).permute(0, 2, 3, 1)
        y = F.interpolate(y, size=[H, W], mode='bilinear', align_corners=True)
        return x*y

class AttBlock(nn.Module):
    def __init__(self, dim, ffn_scale=2.0):
        super().__init__()

        self.norm1 = LayerNorm(dim) 
        self.norm2 = LayerNorm(dim)

        self.gobal = Gobal(dim)

        # Feedforward layer
        self.fc = FC(dim, ffn_scale) 

    def forward(self, x):
        y = self.norm1(x)

        y = self.gobal(y)


        y = self.fc(self.norm2(y)) + y
        return y

###################################
class Mlp(nn.Module):
    def __init__(self, in_features, hidden_features=None, out_features=None, drop=0.):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Conv2d(in_features, hidden_features, 1)
        self.act = nn.GELU()
        self.fc2 = nn.Conv2d(hidden_features, out_features, 1)
        self.drop = nn.Dropout(drop)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x

class HireMLP_w_wai(nn.Module):
    def __init__(self, dim, attn_drop=0., proj_drop=0., pixel=2,
                 step=1, step_pad_mode='c', pixel_pad_mode='c'):
        super().__init__()
        """
        self.pixel: h and w in inner-region rearrangement
        self.step: s in cross-region rearrangement
        """
        # self.pixel = pixel
        self.step = step
        self.step_pad_mode = step_pad_mode
        # self.pixel_pad_mode = pixel_pad_mode
        # print('pixel: {} pad mode: {} step: {} pad mode: {}'.format(
        #     pixel, pixel_pad_mode, step, step_pad_mode))

        # self.mlp_h1 = nn.Conv2d(dim * pixel, dim // 2, 1, bias=False)
        # self.mlp_h1_norm = nn.BatchNorm2d(dim // 2)
        # self.mlp_h2 = nn.Conv2d(dim // 2, dim * pixel, 1, bias=True)
        self.mlp_w1 = nn.Conv2d(dim , dim // 2, 1, bias=False)
        self.mlp_w1_norm = nn.BatchNorm2d(dim // 2)
        self.mlp_w2 = nn.Conv2d(dim // 2, dim, 1, bias=True)
        # self.mlp_c = nn.Conv2d(dim, dim, 1, bias=True)

        self.act = nn.ReLU()

        self.reweight = Mlp(dim, dim // 4, dim * 3)

        self.proj = nn.Conv2d(dim, dim, 1)
        self.proj_drop = nn.Dropout(proj_drop)

    def forward(self, x):
        """
        h: H x W x C -> H/pixel x W x C*pixel
        w: H x W x C -> H x W/pixel x C*pixel
        Setting of F.pad: (left, right, top, bottom)
        """
        B, C, H, W = x.shape
        #print(W)
        # pad_w = (self.pixel - W % self.pixel) % self.pixel
        w = x.clone()

        if self.step:
            if 1:
            #     h = F.pad(h, (0, 0, self.step, 0), "constant", 0)
            #     w = F.pad(w, (self.step, 0, 0, 0), "constant", 0)
            #     h = torch.narrow(h, 2, 0, H)
            #     w = torch.narrow(w, 3, 0, W)
            # elif self.step_pad_mode == 'c':
                #h = torch.roll(h, self.step, -2)
                w = torch.roll(w, self.step, -1)
                #print(w.shape)
                # h = F.pad(h, (0, 0, self.step, 0), mode='circular')
                # w = F.pad(w, (self.step, 0, 0, 0), mode='circular')
            # else:
            #     raise NotImplementedError("Invalid pad mode.")

        # if 1:
        # #     h = F.pad(h, (0, 0, 0, pad_h), "constant", 0)
        # #     w = F.pad(w, (0, pad_w, 0, 0), "constant", 0)
        # # elif self.pixel_pad_mode == 'c':
        #     #h = F.pad(h, (0, 0, 0, pad_h), mode='circular')
        #     w = F.pad(w, (0, pad_w, 0, 0), mode='circular')
        #     print(w.shape)
        # # elif self.pixel_pad_mode == 'replicate':
        # #     h = F.pad(h, (0, 0, 0, pad_h), mode='replicate')
        # #     w = F.pad(w, (0, pad_w, 0, 0), mode='replicate')
        # # else:
        # #     raise NotImplementedError("Invalid pad mode.")
        #
        # # h = h.reshape(B, C, (H + pad_h) // self.pixel, self.pixel, W).permute(0, 1, 3, 2, 4).reshape(B, C * self.pixel,
        # #                                                                                              (
        # #                                                                                                          H + pad_h) // self.pixel,
        # #                                                                                              W)
        # w = w.reshape(B, C, H, (W + pad_w) // self.pixel, self.pixel).permute(0, 1, 4, 2, 3).reshape(B, C * self.pixel,
        #                                                                                              H, (
        #                                                                                                          W + pad_w) // self.pixel)
        # print(w.shape)

        # h = self.mlp_h1(h)
        # h = self.mlp_h1_norm(h)
        # h = self.act(h)
        # h = self.mlp_h2(h)

        w = self.mlp_w1(w)
        w = self.mlp_w1_norm(w)
        w = self.act(w)
        w = self.mlp_w2(w)

        # # h = h.reshape(B, C, self.pixel, (H + pad_h) // self.pixel, W).permute(0, 1, 3, 2, 4).reshape(B, C, H + pad_h, W)
        # w = w.reshape(B, C, self.pixel, H, (W + pad_w) // self.pixel).permute(0, 1, 3, 4, 2).reshape(B, C, H, W + pad_w)
        #
        # # h = torch.narrow(h, 2, 0, H)
        # w = torch.narrow(w, 3, 0, W)

        # cross-region arrangement operation
        if self.step and self.step_pad_mode == 'c':
            # h = torch.roll(h, -self.step, -2)
            w = torch.roll(w, -self.step, -1)
            #print("aaaaaaaa")
            # h = F.pad(h, (0, 0, 0, self.step), mode='circular')
            # w = F.pad(w, (0, self.step, 0, 0), mode='circular')
            # h = torch.narrow(h, 2, self.step, H)
            # w = torch.narrow(w, 3, self.step, W)

        # c = self.mlp_c(x)

        a = (w).flatten(2).mean(2).unsqueeze(2).unsqueeze(2)
        a = self.reweight(a).reshape(B, C, 3).permute(2, 0, 1).softmax(dim=0).unsqueeze(3).unsqueeze(3)

        x = w * a[1]

        x = self.proj(x)
        x = self.proj_drop(x)

        return x

class HireMLP_H_wai(nn.Module):
    def __init__(self, dim, attn_drop=0., proj_drop=0., pixel=2,
                 step=1, step_pad_mode='c', pixel_pad_mode='c'):
        super().__init__()
        """
        self.pixel: h and w in inner-region rearrangement
        self.step: s in cross-region rearrangement
        """

        self.step = step
        self.step_pad_mode = step_pad_mode



        self.mlp_h1 = nn.Conv2d(dim , dim // 2, 1, bias=False)
        self.mlp_h1_norm = nn.BatchNorm2d(dim // 2)
        self.mlp_h2 = nn.Conv2d(dim // 2, dim, 1, bias=True)

        self.act = nn.ReLU()

        self.reweight = Mlp(dim, dim // 4, dim * 3)

        self.proj = nn.Conv2d(dim, dim, 1)
        self.proj_drop = nn.Dropout(proj_drop)

    def forward(self, x):
        """
        h: H x W x C -> H/pixel x W x C*pixel
        w: H x W x C -> H x W/pixel x C*pixel
        Setting of F.pad: (left, right, top, bottom)
        """
        B, C, H, W = x.shape

        # pad_h= (self.pixel - H % self.pixel) % self.pixel
        h= x.clone()

        if self.step:
            if 1:
            #     h = F.pad(h, (0, 0, self.step, 0), "constant", 0)
            #     w = F.pad(w, (self.step, 0, 0, 0), "constant", 0)
            #     h = torch.narrow(h, 2, 0, H)
            #     w = torch.narrow(w, 3, 0, W)
            # elif self.step_pad_mode == 'c':
                h = torch.roll(h, self.step, -2)

                # h = F.pad(h, (0, 0, self.step, 0), mode='circular')
                # w = F.pad(w, (self.step, 0, 0, 0), mode='circular')
            # else:
            #     raise NotImplementedError("Invalid pad mode.")



        h = self.mlp_h1(h)
        h = self.mlp_h1_norm(h)
        h = self.act(h)
        h = self.mlp_h2(h)

        #
        #
        # h = h.reshape(B, C, self.pixel, (H + pad_h) // self.pixel, W).permute(0, 1, 3, 2, 4).reshape(B, C, H + pad_h, W)
        #
        #
        # h = torch.narrow(h, 2, 0, H)
        #

        # cross-region arrangement operation
        if self.step and self.step_pad_mode == 'c':
            h = torch.roll(h, -self.step, -2)

            # h = F.pad(h, (0, 0, 0, self.step), mode='circular')
            # w = F.pad(w, (0, self.step, 0, 0), mode='circular')
            # h = torch.narrow(h, 2, self.step, H)
            # w = torch.narrow(w, 3, self.step, W)



        a = (h ).flatten(2).mean(2).unsqueeze(2).unsqueeze(2)
        a = self.reweight(a).reshape(B, C, 3).permute(2, 0, 1).softmax(dim=0).unsqueeze(3).unsqueeze(3)

        x = h * a[0]

        x = self.proj(x)
        x = self.proj_drop(x)

        return x

class FusionLayer(nn.Module):
    def __init__(self, channel_in, channel_out):
        super(FusionLayer, self).__init__()
        self.weight1 = nn.Parameter(torch.ones(1, channel_in, 1, 1) * 0.5)
        self.weight2 = nn.Parameter(torch.ones(1, channel_in, 1, 1) * 0.5)
        self.expand_conv = nn.Conv2d(channel_in, channel_out, kernel_size=1)  # ?channel_in???channel_out

    def forward(self, x1, x2):
        x = x1 * self.weight1 + x2 * self.weight2
        x = self.expand_conv(x)  # ??1x1??????
        return x

# ??AttBlock_b???????????
class AttBlock_b(nn.Module):
    def __init__(self, dim, ffn_scale=2.0):
        super().__init__()
        self.norm1 = LayerNorm(dim)
        self.norm2 = LayerNorm(dim)
        self.gobal1 = HireMLP_w_wai(dim)
        self.gobal2 = HireMLP_H_wai(dim)
        self.fusion = FusionLayer(dim, dim*2)  # ?????dim*2?reduce_chan???????
        self.reduce_chan = nn.Conv2d(dim*2, dim, 3, 1, 1)  # ???dim*2???FusionLayer????????
        self.fc = FC(dim, ffn_scale)

    def forward(self, x):
        y = self.norm1(x)
        y_1 = self.gobal1(y)
        y_2 = self.gobal2(y)
        y = self.fusion(y_1, y_2)  # ???????
        y = self.reduce_chan(y)  # ??????
        y = self.fc(self.norm2(y)) + y
        return y


###################################

    
class Downsample(nn.Module):
    def __init__(self, n_feat):
        super(Downsample, self).__init__()

        self.body = nn.Sequential(nn.Conv2d(n_feat, n_feat // 2, kernel_size=3, stride=1, padding=1, bias=False),
                                  nn.PixelUnshuffle(2))

    def forward(self, x):
        return self.body(x)

class Upsample(nn.Module):
    def __init__(self, n_feat):
        super(Upsample, self).__init__()

        self.body = nn.Sequential(nn.Conv2d(n_feat, n_feat * 2, kernel_size=3, stride=1, padding=1, bias=False),
                                  nn.PixelShuffle(2))

    def forward(self, x):
        return self.body(x)


class SAFMN(nn.Module):
    def __init__(self, dim, n_blocks=8, ffn_scale=2.0, upscaling_factor=2):
        super().__init__()

        self.to_feat1=nn.Conv2d(3, dim // 4, 3, 1, 1)
        self.to_feat2=nn.PixelUnshuffle(upscaling_factor)
        out_dim = upscaling_factor * dim

        self.feats1 = AttBlock(dim, ffn_scale)
        self.feats2 = AttBlock_b(dim, ffn_scale)
        self.down1_2 = Downsample(dim*1)
        self.feats3 = AttBlock(out_dim*1, ffn_scale)
        self.feats4 = AttBlock_b(out_dim*1, ffn_scale)
        self.feats6 = AttBlock(out_dim*1, ffn_scale)
        self.feats7 = AttBlock_b(out_dim*1, ffn_scale)
        self.up2_1 = Upsample(int(dim*2))
        self.feats8 = AttBlock(dim, ffn_scale)
        self.feats9 = AttBlock_b(dim, ffn_scale)
        self.to_img1=nn.Conv2d(dim, 48, 3, 1, 1)
        self.to_img2=nn.PixelShuffle(4)

        self.reduce_chan_level2 = nn.Conv2d(int(dim * 2), int(dim), kernel_size=1, bias=False)


    def forward(self, x):
        x = F.interpolate(x, scale_factor=1/2, mode='bicubic', align_corners=False)
        x = self.to_feat1(x)
        x = self.to_feat2(x)
        x1 =x
        x = self.feats1(x)
        x = self.feats2(x)
        x_skip = x
        x = self.down1_2(x)
        x = self.feats3(x)
        x = self.feats4(x)
        x = self.feats6(x)
        x = self.feats7(x)
        x = self.up2_1(x)
        x = torch.cat([x,x_skip],1)
        x = self.reduce_chan_level2(x)
        x = self.feats8(x)
        x = self.feats9(x)
        x = self.to_img1(x+x1)
        x = self.to_img2(x)

        return x


if __name__ == '__main__':
    input = torch.rand(1, 3, 256, 256)
    model = SAFMN(dim=64, n_blocks=8, ffn_scale=2.0)
    output = model(input)
    print(output.shape)

    # from fvcore.nn import FlopCountAnalysis, parameter_count_table
    #
    # flops = FlopCountAnalysis(model, input)
    #
    # print("FLOPs: ", flops.total())
    # print(parameter_count_table(model))