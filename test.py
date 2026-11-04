import torch
import cv2
import os
import numpy as np
import time
import sys
import torch.nn.functional as F
from tqdm import tqdm
import re

from model import SharedEncoder, FusionDecoder
from DGCFM import DGCFM
from GTSRM import GTSRM
from models.PURM import PURM
from models.network_ffdnet import FFDNet as net

from AHSM import AHSM
from option import args

CPU_DEVICE = torch.device("cpu")
GPU_DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def load_all_models():
    shared_encoder = SharedEncoder().to(GPU_DEVICE).eval()
    fusion_decoder = FusionDecoder().to(GPU_DEVICE).eval()
    dgcfm_module = DGCFM(hidden_dim=96).to(GPU_DEVICE).eval()

    gtsrm_module = GTSRM(
        img_size=args.patch_size, embed_dim=96, window_size=args.Window_size, num_heads=8, num_stages=4
    ).to(GPU_DEVICE).eval()

    ahsm_module = AHSM().to(CPU_DEVICE).eval()
    purm_model = PURM().to(CPU_DEVICE).eval()
    denoise = net(in_nc=1, out_nc=1, nc=64, nb=15, act_mode='R').to(CPU_DEVICE).eval()

    ahsm_model_path = './model/beta_model_latest.pth'
    ahsm_module.load_state_dict(torch.load(ahsm_model_path, map_location='cpu'), strict=False)

    purm_model.load_state_dict(torch.load(args.GC_model, map_location='cpu')['model'])
    denoise.load_state_dict(torch.load(args.Denoise_model, map_location='cpu'), strict=True)

    current_dir = os.path.dirname(os.path.abspath(__file__))
    model_file = os.path.join(current_dir, 'checkpoints', 'best_model.pth')

    if not os.path.exists(model_file):
        sys.exit(1)

    ckpt = torch.load(model_file, map_location='cpu')
    shared_encoder.load_state_dict(ckpt['shared_encoder'])
    fusion_decoder.load_state_dict(ckpt['fusion_decoder'])
    dgcfm_module.load_state_dict(ckpt['dgcfm'])
    gtsrm_module.load_state_dict(ckpt['gtsrm'])

    return {
        'shared_encoder': shared_encoder,
        'fusion_decoder': fusion_decoder,
        'ahsm': ahsm_module,
        'dgcfm': dgcfm_module,
        'gtsrm': gtsrm_module,
        'purm': purm_model,
        'denoise': denoise
    }


def load_image(img_path):
    img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
    if img is None:
        raise ValueError(f"Cannot load image: {img_path}")
    img = img.astype(np.float32) / 255.0
    return torch.from_numpy(img).unsqueeze(0).unsqueeze(0).float()


def save_image(tensor, save_path):
    img = tensor.squeeze().cpu().numpy()
    img = np.clip(img * 255.0, 0, 255).astype(np.uint8)
    cv2.imwrite(save_path, img)


def create_hanning_window(window_size, device):
    win_1d = torch.hann_window(window_size, periodic=False, device=device)
    win_2d = win_1d.unsqueeze(1) * win_1d.unsqueeze(0)
    return win_2d.unsqueeze(0).unsqueeze(0)


def process_one_pair(over_path, under_path, models, save_name, output_dir):
    noise_level = 10
    patch_size = args.patch_size
    stride = patch_size // 2

    original_img1 = load_image(over_path)
    original_img6 = load_image(under_path)
    _, _, H_orig, W_orig = original_img1.shape

    with torch.no_grad():
        pad_h = (stride - H_orig % stride) % stride
        pad_w = (stride - W_orig % stride) % stride

        padding = (stride, stride + pad_w, stride, stride + pad_h)

        p_img1 = F.pad(original_img1, padding, mode='replicate')
        p_img6 = F.pad(original_img6, padding, mode='replicate')

        p_enh_o = models['ahsm'](p_img1)
        p_enh_u, _, _, _ = models['purm'](p_img6)

        sigma = torch.full((1, 1, 1, 1), noise_level / 255.0, device=CPU_DEVICE).type_as(p_enh_u)
        p_enh_u = models['denoise'](p_enh_u, sigma)

        _, _, H_pad, W_pad = p_img1.shape

        result = torch.zeros(1, 1, H_pad, W_pad, device=CPU_DEVICE)
        weight_map = torch.zeros(1, 1, H_pad, W_pad, device=CPU_DEVICE)

        window = create_hanning_window(patch_size, GPU_DEVICE)

        ys = list(range(0, H_pad - patch_size + 1, stride))
        xs = list(range(0, W_pad - patch_size + 1, stride))

        for y in ys:
            for x in xs:
                p1 = p_img1[:, :, y:y + patch_size, x:x + patch_size].to(GPU_DEVICE)
                p6 = p_img6[:, :, y:y + patch_size, x:x + patch_size].to(GPU_DEVICE)
                peo = p_enh_o[:, :, y:y + patch_size, x:x + patch_size].to(GPU_DEVICE)
                peu = p_enh_u[:, :, y:y + patch_size, x:x + patch_size].to(GPU_DEVICE)

                fo = models['shared_encoder'](p1)
                fu = models['shared_encoder'](p6)
                fe_o = models['shared_encoder'](peo)
                fe_u = models['shared_encoder'](peu)

                C_fused = models['dgcfm'](fo, fu)
                fused_feat = models['gtsrm'](fe_o, fe_u, C_fused)
                out = models['fusion_decoder'](fused_feat)

                out_weighted = out * window

                result[:, :, y:y + patch_size, x:x + patch_size] += out_weighted.to(CPU_DEVICE)
                weight_map[:, :, y:y + patch_size, x:x + patch_size] += window.to(CPU_DEVICE)

        result = result / (weight_map + 1e-8)
        result = result[:, :, stride: stride + H_orig, stride: stride + W_orig]

    save_image(result, os.path.join(output_dir, f"{save_name}.png"))
    return result


def natural_sort_key(s):
    return [int(text) if text.isdigit() else text.lower() for text in re.split(r'(\d+)', s)]


def run_test():
    testA = './dataset/CW-MEF/test_data/over'
    testB = './dataset/CW-MEF/test_data/under'
    output = './output/test_result'

    os.makedirs(output, exist_ok=True)

    imgs_over = sorted([f for f in os.listdir(testA) if f.lower().endswith(('.png', '.jpg'))], key=natural_sort_key)
    imgs_under = sorted([f for f in os.listdir(testB) if f.lower().endswith(('.png', '.jpg'))], key=natural_sort_key)

    if not imgs_over or not imgs_under:
        sys.exit(1)

    models = load_all_models()
    total_t0 = time.time()

    for i, (name_over, name_under) in enumerate(
            tqdm(zip(imgs_over, imgs_under), total=len(imgs_over), desc="Processing", ncols=100)):
        path_over = os.path.join(testA, name_over)
        path_under = os.path.join(testB, name_under)
        save_name = os.path.splitext(name_over)[0]

        try:
            process_one_pair(path_over, path_under, models, save_name, output)
        except Exception as e:
            continue

    total_time = time.time() - total_t0
    print(f"Done. Time: {total_time / 60:.2f} min")


if __name__ == '__main__':
    run_test()