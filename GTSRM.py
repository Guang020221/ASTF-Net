# coding: utf-8
import torch
import torch.nn as nn
import torch.nn.functional as F
from timm.layers import to_2tuple, trunc_normal_


def window_partition(x, window_size):
    B, H, W, C = x.shape
    x = x.view(B, H // window_size, window_size,
               W // window_size, window_size, C)
    windows = x.permute(0, 1, 3, 2, 4, 5).contiguous()
    windows = windows.view(-1, window_size, window_size, C)
    return windows


def window_reverse(windows, window_size, H, W):
    B = int(windows.shape[0] / (H * W / window_size / window_size))
    x = windows.view(B, H // window_size, W // window_size,
                     window_size, window_size, -1)
    x = x.permute(0, 1, 3, 2, 4, 5).contiguous().view(B, H, W, -1)
    return x


class PatchEmbed(nn.Module):
    def __init__(self, img_size=224, patch_size=1, in_chans=96,
                 embed_dim=96, norm_layer=None):
        super().__init__()
        img_size = to_2tuple(img_size)
        patch_size = to_2tuple(patch_size)
        patches_resolution = [img_size[0] // patch_size[0],
                              img_size[1] // patch_size[1]]
        self.img_size = img_size
        self.patch_size = patch_size
        self.patches_resolution = patches_resolution
        self.num_patches = patches_resolution[0] * patches_resolution[1]
        self.in_chans = in_chans
        self.embed_dim = embed_dim
        self.norm = norm_layer(embed_dim) if norm_layer is not None else None

    def forward(self, x):
        x = x.flatten(2).transpose(1, 2)
        if self.norm is not None:
            x = self.norm(x)
        return x


class PatchUnEmbed(nn.Module):
    def __init__(self, img_size=224, patch_size=1, in_chans=96,
                 embed_dim=96, norm_layer=None):
        super().__init__()
        img_size = to_2tuple(img_size)
        patch_size = to_2tuple(patch_size)
        patches_resolution = [img_size[0] // patch_size[0],
                              img_size[1] // patch_size[1]]
        self.img_size = img_size
        self.patch_size = patch_size
        self.patches_resolution = patches_resolution
        self.num_patches = patches_resolution[0] * patches_resolution[1]
        self.in_chans = in_chans
        self.embed_dim = embed_dim

    def forward(self, x, x_size):
        B, HW, C = x.shape
        x = x.transpose(1, 2).view(B, self.embed_dim, x_size[0], x_size[1])
        return x


class Mlp(nn.Module):
    def __init__(self, in_features, hidden_features=None, out_features=None,
                 act_layer=nn.GELU, drop=0.):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = act_layer()
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.drop = nn.Dropout(drop)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x


class C_WindowAttention(nn.Module):
    def __init__(self, dim, window_size, num_heads, qkv_bias=True,
                 qk_scale=None, attn_drop=0., proj_drop=0.):
        super().__init__()
        self.dim = dim
        self.window_size = window_size
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = qk_scale or head_dim ** -0.5

        self.relative_position_bias_table = nn.Parameter(
            torch.zeros((2 * window_size[0] - 1) * (2 * window_size[1] - 1), num_heads))

        coords_h = torch.arange(self.window_size[0])
        coords_w = torch.arange(self.window_size[1])
        coords = torch.stack(torch.meshgrid([coords_h, coords_w], indexing='ij'))
        coords_flatten = torch.flatten(coords, 1)
        relative_coords = coords_flatten[:, :, None] - coords_flatten[:, None, :]
        relative_coords = relative_coords.permute(1, 2, 0).contiguous()
        relative_coords[:, :, 0] += self.window_size[0] - 1
        relative_coords[:, :, 1] += self.window_size[1] - 1
        relative_coords[:, :, 0] *= 2 * self.window_size[1] - 1
        relative_position_index = relative_coords.sum(-1)
        self.register_buffer("relative_position_index", relative_position_index)

        self.q = nn.Linear(dim, dim, bias=qkv_bias)
        self.kv = nn.Linear(dim, dim * 2, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)
        trunc_normal_(self.relative_position_bias_table, std=.02)
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, x, y, mask=None):
        B_, N, C = x.shape
        q = self.q(x).reshape(B_, N, 1, self.num_heads,
                              C // self.num_heads).permute(2, 0, 3, 1, 4)
        kv = self.kv(y).reshape(B_, N, 2, self.num_heads,
                                C // self.num_heads).permute(2, 0, 3, 1, 4)
        q, k, v = q[0], kv[0], kv[1]
        q = q * self.scale
        attn = (q @ k.transpose(-2, -1))

        relative_position_bias = self.relative_position_bias_table[
            self.relative_position_index.view(-1)].view(
            self.window_size[0] * self.window_size[1],
            self.window_size[0] * self.window_size[1], -1)
        relative_position_bias = relative_position_bias.permute(2, 0, 1).contiguous()
        attn = attn + relative_position_bias.unsqueeze(0)

        if mask is not None:
            nW = mask.shape[0]
            attn = attn.view(B_ // nW, nW, self.num_heads, N, N) + \
                   mask.unsqueeze(1).unsqueeze(0).to(x.device)
            attn = attn.view(-1, self.num_heads, N, N)
            attn = self.softmax(attn)
        else:
            attn = self.softmax(attn)

        attn = self.attn_drop(attn)
        x = (attn @ v).transpose(1, 2).reshape(B_, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x


class C_WindowAttention_Triple(nn.Module):
    def __init__(self, dim, window_size, num_heads, qkv_bias=True,
                 qk_scale=None, attn_drop=0., proj_drop=0.):
        super().__init__()
        self.dim = dim
        self.window_size = window_size
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = qk_scale or head_dim ** -0.5

        self.relative_position_bias_table = nn.Parameter(
            torch.zeros((2 * window_size[0] - 1) * (2 * window_size[1] - 1), num_heads))

        coords_h = torch.arange(self.window_size[0])
        coords_w = torch.arange(self.window_size[1])
        coords = torch.stack(torch.meshgrid([coords_h, coords_w], indexing='ij'))
        coords_flatten = torch.flatten(coords, 1)
        relative_coords = coords_flatten[:, :, None] - coords_flatten[:, None, :]
        relative_coords = relative_coords.permute(1, 2, 0).contiguous()
        relative_coords[:, :, 0] += self.window_size[0] - 1
        relative_coords[:, :, 1] += self.window_size[1] - 1
        relative_coords[:, :, 0] *= 2 * self.window_size[1] - 1
        relative_position_index = relative_coords.sum(-1)
        self.register_buffer("relative_position_index", relative_position_index)

        self.q = nn.Linear(dim, dim, bias=qkv_bias)
        self.kv = nn.Linear(dim, dim * 2, bias=qkv_bias)

        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)
        trunc_normal_(self.relative_position_bias_table, std=.02)
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, x, y1, y2, mask=None):
        B_, N, C = x.shape
        q = self.q(x).reshape(B_, N, 1, self.num_heads,
                              C // self.num_heads).permute(2, 0, 3, 1, 4)
        q = q[0]

        kv1 = self.kv(y1).reshape(B_, N, 2, self.num_heads,
                                  C // self.num_heads).permute(2, 0, 3, 1, 4)
        k1, v1 = kv1[0], kv1[1]

        kv2 = self.kv(y2).reshape(B_, N, 2, self.num_heads,
                                  C // self.num_heads).permute(2, 0, 3, 1, 4)
        k2, v2 = kv2[0], kv2[1]

        k = torch.cat([k1, k2], dim=2)
        v = torch.cat([v1, v2], dim=2)

        q = q * self.scale
        attn = (q @ k.transpose(-2, -1))

        relative_position_bias = self.relative_position_bias_table[
            self.relative_position_index.view(-1)].view(
            self.window_size[0] * self.window_size[1],
            self.window_size[0] * self.window_size[1], -1)
        relative_position_bias = relative_position_bias.permute(2, 0, 1).contiguous()

        relative_position_bias = torch.cat([relative_position_bias, relative_position_bias], dim=-1)
        attn = attn + relative_position_bias.unsqueeze(0)

        if mask is not None:
            nW = mask.shape[0]
            mask_expanded = torch.cat([mask, mask], dim=-1)
            attn = attn.view(B_ // nW, nW, self.num_heads, N, 2 * N) + \
                   mask_expanded.unsqueeze(1).unsqueeze(0).to(x.device)
            attn = attn.view(-1, self.num_heads, N, 2 * N)
            attn = self.softmax(attn)
        else:
            attn = self.softmax(attn)

        attn = self.attn_drop(attn)
        x = (attn @ v).transpose(1, 2).reshape(B_, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x


class CSTB(nn.Module):


    def __init__(self, dim, input_resolution, num_heads, window_size=8,
                 shift_size=0, mlp_ratio=4., qkv_bias=True, qk_scale=None,
                 drop=0., attn_drop=0., act_layer=nn.GELU, norm_layer=nn.LayerNorm):
        super().__init__()
        self.dim = dim
        self.input_resolution = input_resolution
        self.num_heads = num_heads
        self.window_size = window_size
        self.shift_size = shift_size
        self.mlp_ratio = mlp_ratio

        if min(self.input_resolution) <= self.window_size:
            self.shift_size = 0
            self.window_size = min(self.input_resolution)
        assert 0 <= self.shift_size < self.window_size


        self.norm1_A = norm_layer(dim)
        self.norm1_B = norm_layer(dim)
        self.attn_A = C_WindowAttention(
            dim, window_size=to_2tuple(self.window_size), num_heads=num_heads,
            qkv_bias=qkv_bias, qk_scale=qk_scale,
            attn_drop=attn_drop, proj_drop=drop)
        self.attn_B = C_WindowAttention(
            dim, window_size=to_2tuple(self.window_size), num_heads=num_heads,
            qkv_bias=qkv_bias, qk_scale=qk_scale,
            attn_drop=attn_drop, proj_drop=drop)

        self.drop_path_A = nn.Identity()
        self.drop_path_B = nn.Identity()
        self.norm2_A = norm_layer(dim)
        self.norm2_B = norm_layer(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp_A = Mlp(in_features=dim, hidden_features=mlp_hidden_dim,
                         act_layer=act_layer, drop=drop)
        self.mlp_B = Mlp(in_features=dim, hidden_features=mlp_hidden_dim,
                         act_layer=act_layer, drop=drop)

        if self.shift_size > 0:
            self.register_buffer("attn_mask", self.cal_mask(self.input_resolution))
        else:
            self.register_buffer("attn_mask", None)

    def cal_mask(self, x_size):
        H, W = x_size
        img_mask = torch.zeros((1, H, W, 1))
        h_slices = (slice(0, -self.window_size),
                    slice(-self.window_size, -self.shift_size),
                    slice(-self.shift_size, None))
        w_slices = (slice(0, -self.window_size),
                    slice(-self.window_size, -self.shift_size),
                    slice(-self.shift_size, None))
        cnt = 0
        for h in h_slices:
            for w in w_slices:
                img_mask[:, h, w, :] = cnt
                cnt += 1
        mask_windows = window_partition(img_mask, self.window_size)
        mask_windows = mask_windows.view(-1, self.window_size * self.window_size)
        attn_mask = mask_windows.unsqueeze(1) - mask_windows.unsqueeze(2)
        attn_mask = attn_mask.masked_fill(attn_mask != 0, float(-100.0)) \
            .masked_fill(attn_mask == 0, float(0.0))
        return attn_mask

    def forward(self, x_s, y_s, x_size):
        x = x_s
        y = y_s

        H, W = x_size
        B, L, C = x.shape
        shortcut_A = x
        shortcut_B = y

        x = self.norm1_A(x)
        y = self.norm1_B(y)
        x = x.view(B, H, W, C)
        y = y.view(B, H, W, C)

        if self.shift_size > 0:
            shifted_x = torch.roll(x, shifts=(-self.shift_size, -self.shift_size), dims=(1, 2))
            shifted_y = torch.roll(y, shifts=(-self.shift_size, -self.shift_size), dims=(1, 2))
        else:
            shifted_x = x
            shifted_y = y

        x_windows = window_partition(shifted_x, self.window_size).view(
            -1, self.window_size * self.window_size, C)
        y_windows = window_partition(shifted_y, self.window_size).view(
            -1, self.window_size * self.window_size, C)

        if self.input_resolution == x_size:
            mask = self.attn_mask
        else:
            mask = self.cal_mask(x_size).to(x.device) if self.shift_size > 0 else None

        attn_windows_A = self.attn_A(x_windows, y_windows, mask=mask)
        attn_windows_B = self.attn_B(y_windows, x_windows, mask=mask)

        attn_windows_A = attn_windows_A.view(-1, self.window_size, self.window_size, C)
        attn_windows_B = attn_windows_B.view(-1, self.window_size, self.window_size, C)
        shifted_x = window_reverse(attn_windows_A, self.window_size, H, W)
        shifted_y = window_reverse(attn_windows_B, self.window_size, H, W)

        if self.shift_size > 0:
            x = torch.roll(shifted_x, shifts=(self.shift_size, self.shift_size), dims=(1, 2))
            y = torch.roll(shifted_y, shifts=(self.shift_size, self.shift_size), dims=(1, 2))
        else:
            x = shifted_x
            y = shifted_y

        x = x.view(B, H * W, C)
        y = y.view(B, H * W, C)
        x = shortcut_A + self.drop_path_A(x)
        x = x + self.drop_path_A(self.mlp_A(self.norm2_A(x)))
        y = shortcut_B + self.drop_path_B(y)
        y = y + self.drop_path_B(self.mlp_B(self.norm2_B(y)))
        return x, y


class TSTB(nn.Module):


    def __init__(self, dim, input_resolution, num_heads, window_size=8,
                 shift_size=0, mlp_ratio=4., qkv_bias=True, qk_scale=None,
                 drop=0., attn_drop=0., act_layer=nn.GELU, norm_layer=nn.LayerNorm):
        super().__init__()
        self.dim = dim
        self.input_resolution = input_resolution
        self.num_heads = num_heads
        self.window_size = window_size
        self.shift_size = shift_size
        self.mlp_ratio = mlp_ratio

        if min(self.input_resolution) <= self.window_size:
            self.shift_size = 0
            self.window_size = min(self.input_resolution)
        assert 0 <= self.shift_size < self.window_size

        self.cstb = CSTB(
            dim=dim, input_resolution=input_resolution, num_heads=num_heads,
            window_size=window_size, shift_size=self.shift_size,
            mlp_ratio=mlp_ratio, qkv_bias=qkv_bias, qk_scale=qk_scale,
            drop=drop, attn_drop=attn_drop, norm_layer=norm_layer)


        self.norm1_fused = norm_layer(dim)
        self.norm1_over = norm_layer(dim)
        self.norm1_under = norm_layer(dim)
        self.attn_triple = C_WindowAttention_Triple(
            dim, window_size=to_2tuple(self.window_size), num_heads=num_heads,
            qkv_bias=qkv_bias, qk_scale=qk_scale,
            attn_drop=attn_drop, proj_drop=drop)

        self.drop_path_fused = nn.Identity()
        self.norm2_fused = norm_layer(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp_fused = Mlp(in_features=dim, hidden_features=mlp_hidden_dim,
                             act_layer=act_layer, drop=drop)

        if self.shift_size > 0:
            self.register_buffer("attn_mask", self.cal_mask(self.input_resolution))
        else:
            self.register_buffer("attn_mask", None)

    def cal_mask(self, x_size):
        H, W = x_size
        img_mask = torch.zeros((1, H, W, 1))
        h_slices = (slice(0, -self.window_size),
                    slice(-self.window_size, -self.shift_size),
                    slice(-self.shift_size, None))
        w_slices = (slice(0, -self.window_size),
                    slice(-self.window_size, -self.shift_size),
                    slice(-self.shift_size, None))
        cnt = 0
        for h in h_slices:
            for w in w_slices:
                img_mask[:, h, w, :] = cnt
                cnt += 1
        mask_windows = window_partition(img_mask, self.window_size)
        mask_windows = mask_windows.view(-1, self.window_size * self.window_size)
        attn_mask = mask_windows.unsqueeze(1) - mask_windows.unsqueeze(2)
        attn_mask = attn_mask.masked_fill(attn_mask != 0, float(-100.0)) \
            .masked_fill(attn_mask == 0, float(0.0))
        return attn_mask

    def forward(self, C_over, C_under, C_fused, x_size):
        H, W = x_size
        B, L, C = C_over.shape

        C_over, C_under = self.cstb(C_over, C_under, x_size)


        shortcut_fused = C_fused
        x_f = self.norm1_fused(C_fused).view(B, H, W, C)
        x_o = self.norm1_over(C_over).view(B, H, W, C)
        x_u = self.norm1_under(C_under).view(B, H, W, C)

        if self.shift_size > 0:
            x_f = torch.roll(x_f, shifts=(-self.shift_size, -self.shift_size), dims=(1, 2))
            x_o = torch.roll(x_o, shifts=(-self.shift_size, -self.shift_size), dims=(1, 2))
            x_u = torch.roll(x_u, shifts=(-self.shift_size, -self.shift_size), dims=(1, 2))

        f_win = window_partition(x_f, self.window_size).view(
            -1, self.window_size * self.window_size, C)
        o_win = window_partition(x_o, self.window_size).view(
            -1, self.window_size * self.window_size, C)
        u_win = window_partition(x_u, self.window_size).view(
            -1, self.window_size * self.window_size, C)

        if self.input_resolution == x_size:
            mask = self.attn_mask
        else:
            mask = self.cal_mask(x_size).to(C_fused.device) if self.shift_size > 0 else None

        attn_f = self.attn_triple(f_win, o_win, u_win, mask=mask)
        attn_f = attn_f.view(-1, self.window_size, self.window_size, C)
        x_f = window_reverse(attn_f, self.window_size, H, W)

        if self.shift_size > 0:
            x_f = torch.roll(x_f, shifts=(self.shift_size, self.shift_size), dims=(1, 2))

        x_f = x_f.view(B, H * W, C)
        x_f = shortcut_fused + self.drop_path_fused(x_f)
        x_f = x_f + self.drop_path_fused(self.mlp_fused(self.norm2_fused(x_f)))
        C_fused = x_f

        return C_over, C_under, C_fused


class GTSRM(nn.Module):

    def __init__(self, img_size=256, embed_dim=96, num_heads=8, window_size=8,
                 mlp_ratio=4., qkv_bias=True, qk_scale=None,
                 drop_rate=0., attn_drop_rate=0., norm_layer=nn.LayerNorm,
                 num_stages=2):
        super().__init__()
        self.embed_dim = embed_dim
        self.mlp_ratio = mlp_ratio

        self.patch_embed = PatchEmbed(
            img_size=img_size, patch_size=1, in_chans=embed_dim,
            embed_dim=embed_dim, norm_layer=norm_layer)
        patches_resolution = self.patch_embed.patches_resolution
        self.patches_resolution = patches_resolution

        self.patch_unembed = PatchUnEmbed(
            img_size=img_size, patch_size=1, in_chans=embed_dim,
            embed_dim=embed_dim, norm_layer=norm_layer)

        self.norm_layer = norm_layer(embed_dim)
        self.input_norm_over = nn.LayerNorm(embed_dim)
        self.input_norm_under = nn.LayerNorm(embed_dim)
        self.input_norm_fused = nn.LayerNorm(embed_dim)

        self.tstb_list = nn.ModuleList()
        for i in range(num_stages):
            current_shift_size = 0 if (i % 2 == 0) else (window_size // 2)

            self.tstb_list.append(
                TSTB(
                    dim=embed_dim,
                    input_resolution=(patches_resolution[0], patches_resolution[1]),
                    num_heads=num_heads,
                    window_size=window_size,
                    shift_size=current_shift_size,
                    mlp_ratio=mlp_ratio,
                    qkv_bias=qkv_bias,
                    qk_scale=qk_scale,
                    drop=drop_rate,
                    attn_drop=attn_drop_rate,
                    norm_layer=norm_layer
                )
            )

    def forward(self, C_over, C_under, C_fused):
        B, C, H, W = C_over.shape
        x_size = (H, W)

        def preprocess(feat, norm):
            flat = feat.view(B, C, -1).transpose(1, 2)
            flat = norm(flat)
            feat = flat.transpose(1, 2).view(B, C, H, W)
            return self.patch_embed(feat)

        over = preprocess(C_over, self.input_norm_over)
        under = preprocess(C_under, self.input_norm_under)
        fused = preprocess(C_fused, self.input_norm_fused)

        for tstb in self.tstb_list:
            over, under, fused = tstb(over, under, fused, x_size)

        out = self.norm_layer(fused)
        out = self.patch_unembed(out, x_size)
        return out