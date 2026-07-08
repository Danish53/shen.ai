"""DeepPhys — 2D convolutional attention network for rPPG (ECCV 2018)."""
import torch
import torch.nn as nn


class AttentionMask(nn.Module):
    def forward(self, x):
        xsum = torch.sum(x, dim=2, keepdim=True)
        xsum = torch.sum(xsum, dim=3, keepdim=True)
        xshape = tuple(x.size())
        return x / xsum * xshape[2] * xshape[3] * 0.5


class DeepPhys(nn.Module):
    def __init__(
        self,
        in_channels=3,
        nb_filters1=32,
        nb_filters2=64,
        kernel_size=3,
        dropout_rate1=0.25,
        dropout_rate2=0.5,
        pool_size=(2, 2),
        nb_dense=128,
        img_size=72,
    ):
        super().__init__()
        self.motion_conv1 = nn.Conv2d(in_channels, nb_filters1, kernel_size, padding=1, bias=True)
        self.motion_conv2 = nn.Conv2d(nb_filters1, nb_filters1, kernel_size, bias=True)
        self.motion_conv3 = nn.Conv2d(nb_filters1, nb_filters2, kernel_size, padding=1, bias=True)
        self.motion_conv4 = nn.Conv2d(nb_filters2, nb_filters2, kernel_size, bias=True)

        self.apperance_conv1 = nn.Conv2d(in_channels, nb_filters1, kernel_size, padding=1, bias=True)
        self.apperance_conv2 = nn.Conv2d(nb_filters1, nb_filters1, kernel_size, bias=True)
        self.apperance_conv3 = nn.Conv2d(nb_filters1, nb_filters2, kernel_size, padding=1, bias=True)
        self.apperance_conv4 = nn.Conv2d(nb_filters2, nb_filters2, kernel_size, bias=True)

        self.apperance_att_conv1 = nn.Conv2d(nb_filters1, 1, kernel_size=1, bias=True)
        self.attn_mask_1 = AttentionMask()
        self.apperance_att_conv2 = nn.Conv2d(nb_filters2, 1, kernel_size=1, bias=True)
        self.attn_mask_2 = AttentionMask()

        self.avg_pooling_1 = nn.AvgPool2d(pool_size)
        self.avg_pooling_2 = nn.AvgPool2d(pool_size)
        self.avg_pooling_3 = nn.AvgPool2d(pool_size)
        self.dropout_1 = nn.Dropout(dropout_rate1)
        self.dropout_2 = nn.Dropout(dropout_rate1)
        self.dropout_3 = nn.Dropout(dropout_rate1)
        self.dropout_4 = nn.Dropout(dropout_rate2)

        dense_in = {36: 3136, 72: 16384, 96: 30976}.get(img_size)
        if dense_in is None:
            raise ValueError(f"Unsupported image size: {img_size}")
        self.final_dense_1 = nn.Linear(dense_in, nb_dense, bias=True)
        self.final_dense_2 = nn.Linear(nb_dense, 1, bias=True)

    def forward(self, inputs):
        diff_input = inputs[:, :3, :, :]
        raw_input = inputs[:, 3:, :, :]

        d1 = torch.tanh(self.motion_conv1(diff_input))
        d2 = torch.tanh(self.motion_conv2(d1))

        r1 = torch.tanh(self.apperance_conv1(raw_input))
        r2 = torch.tanh(self.apperance_conv2(r1))

        g1 = torch.sigmoid(self.apperance_att_conv1(r2))
        gated1 = d2 * self.attn_mask_1(g1)

        d3 = self.avg_pooling_1(gated1)
        d4 = self.dropout_1(d3)
        r3 = self.avg_pooling_2(r2)
        r4 = self.dropout_2(r3)

        d5 = torch.tanh(self.motion_conv3(d4))
        d6 = torch.tanh(self.motion_conv4(d5))
        r5 = torch.tanh(self.apperance_conv3(r4))
        r6 = torch.tanh(self.apperance_conv4(r5))

        g2 = torch.sigmoid(self.apperance_att_conv2(r6))
        gated2 = d6 * self.attn_mask_2(g2)

        d7 = self.avg_pooling_3(gated2)
        d8 = self.dropout_3(d7)
        d9 = d8.reshape(d8.size(0), -1)
        d10 = torch.tanh(self.final_dense_1(d9))
        d11 = self.dropout_4(d10)
        return self.final_dense_2(d11)
