import torch
from torch import optim
from dataset import Trainset
from models.PURM import PURM
from models.network_ffdnet import FFDNet as net
import os
import numpy as np
import torch.utils.data as data
from option import args
from torchvision import transforms
from tqdm import tqdm
from MEFLoss import fusion_loss_mef
from model import SharedEncoder, FusionDecoder
from DGCFM import DGCFM
from GTSRM import GTSRM

from AHSM import AHSM

import time
import sys
import gc
from vgg import vgg16
from utils import features_grad
import torch.nn.functional as F
from pytorch_msssim import ms_ssim
import random
import cv2


def compute_vgg_weights(feature_model, img_over, img_under, temperature=0.01):
    weights_list = []
    _, _, orig_H, orig_W = img_over.shape

    for img_single in [img_over, img_under]:
        img_3ch = torch.cat((img_single, img_single, img_single), dim=1)
        feats = feature_model(img_3ch)
        w_single = []
        for feat_layer in feats:
            m = torch.mean(features_grad(feat_layer).pow(2), dim=1, keepdim=True)
            m = F.interpolate(m, size=(orig_H, orig_W), mode='bilinear', align_corners=False)
            w_single.append(m)

        del feats, img_3ch

        w_single = torch.cat(w_single, dim=1)
        weight_score = torch.mean(w_single, dim=1, keepdim=True)
        weights_list.append(weight_score)

    weight_list = torch.cat(weights_list, dim=1)
    weight_list = F.softmax(weight_list / temperature, dim=1)
    return weight_list


def main():
    torch.backends.cudnn.benchmark = True
    torch.backends.cudnn.deterministic = False
    torch.cuda.empty_cache()

    LEARNING_RATE = 1e-4
    WEIGHT_DECAY = 1e-5
    BATCH_SIZE = 1
    EPOCHS = args.epochs
    WARMUP_EPOCHS = 5


    transform = transforms.Compose([transforms.ToTensor()])
    train_set = Trainset(transform=transform)
    train_loader = data.DataLoader(
        train_set,
        batch_size=BATCH_SIZE,
        shuffle=True,
        num_workers=4,
        drop_last=True,
        pin_memory=True,
        persistent_workers=True
    )

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    shared_encoder = SharedEncoder().to(device)
    fusion_decoder = FusionDecoder().to(device)
    dgcfm_module = DGCFM(hidden_dim=96).to(device)
    gtsrm_module = GTSRM(
        img_size=args.patch_size,
        embed_dim=96,
        window_size=args.Window_size,
        num_heads=8,
        num_stages=4
    ).to(device)

    ahsm_module = AHSM().to(device)
    ahsm_model_path = './model/beta_model_latest.pth'
    try:
        ahsm_state = torch.load(ahsm_model_path, map_location=device)
        ahsm_module.load_state_dict(ahsm_state, strict=False)
    except Exception as e:
        raise RuntimeError(f"Failed to load AHSM model: {e}")
    for param in ahsm_module.parameters():
        param.requires_grad = False
    ahsm_module.eval()

    purm_model = PURM().to(device)
    try:
        purm_state = torch.load(args.GC_model, map_location=device)
        purm_model.load_state_dict(purm_state['model'])
    except Exception as e:
        raise RuntimeError(f"Failed to load PURM model: {e}")
    for param in purm_model.parameters():
        param.requires_grad = False
    purm_model.eval()

    Denoise_model = net(in_nc=1, out_nc=1, nc=64, nb=15, act_mode='R').to(device)
    try:
        Denoise_state = torch.load(args.Denoise_model, map_location=device)
        Denoise_model.load_state_dict(Denoise_state, strict=True)
    except Exception as e:
        raise RuntimeError(f"Failed to load Denoise model: {e}")
    for param in Denoise_model.parameters():
        param.requires_grad = False
    Denoise_model.eval()

    feature_model = vgg16().to(device)
    try:
        vgg_state = torch.load('vgg16.pth', map_location=device)
        feature_model.load_state_dict(vgg_state)
    except Exception as e:
        raise RuntimeError(f"Failed to load VGG16 model: {e}")
    for param in feature_model.parameters():
        param.requires_grad = False
    feature_model.eval()

    trainable_params = (
            list(shared_encoder.parameters()) +
            list(fusion_decoder.parameters()) +
            list(dgcfm_module.parameters()) +
            list(gtsrm_module.parameters())
    )
    print(f"Total Params: {sum(p.numel() for p in trainable_params) / 1e6:.2f}M")
    optimizer = optim.AdamW(
        trainable_params,
        lr=LEARNING_RATE,
        weight_decay=WEIGHT_DECAY,
        betas=(0.9, 0.999),
        eps=1e-8
    )

    Loss_Fun = fusion_loss_mef(total_epochs=EPOCHS).to(device)

    class WarmupCosineScheduler:
        def __init__(self, optimizer, warmup_epochs, total_epochs, base_lr, max_lr, min_lr):
            self.optimizer = optimizer
            self.warmup_epochs = warmup_epochs
            self.total_epochs = total_epochs
            self.base_lr = base_lr
            self.max_lr = max_lr
            self.min_lr = min_lr
            self.current_epoch = 0

        def step(self):
            if self.current_epoch < self.warmup_epochs:
                lr = self.base_lr + (self.max_lr - self.base_lr) * \
                     (self.current_epoch / self.warmup_epochs)
                stage = "Warmup"
            else:
                progress = (self.current_epoch - self.warmup_epochs) / \
                           (self.total_epochs - self.warmup_epochs)
                lr = self.min_lr + (self.max_lr - self.min_lr) * \
                     0.5 * (1 + np.cos(np.pi * progress))
                stage = "Cosine"
            for param_group in self.optimizer.param_groups:
                param_group['lr'] = lr
            self.current_epoch += 1
            return lr, stage

    scheduler = WarmupCosineScheduler(
        optimizer,
        warmup_epochs=WARMUP_EPOCHS,
        total_epochs=EPOCHS,
        base_lr=2e-5,
        max_lr=LEARNING_RATE,
        min_lr=1e-6
    )

    os.makedirs("output/fused_images", exist_ok=True)
    os.makedirs("output/train_vis", exist_ok=True)

    if args.model_path == '':
        args.model_path = './checkpoints/'
    os.makedirs(args.model_path, exist_ok=True)

    noise_level_img = 10
    Loss_list = []
    best_loss = float('inf')
    best_epoch = 0
    epoch_times = []
    vgg_weight_over_list = []
    vgg_weight_under_list = []

    start_epoch = 0
    if args.resume and os.path.isfile(args.resume):
        checkpoint = torch.load(args.resume, map_location=device)

        shared_encoder.load_state_dict(checkpoint['shared_encoder'])
        fusion_decoder.load_state_dict(checkpoint['fusion_decoder'])
        dgcfm_module.load_state_dict(checkpoint['dgcfm'])
        gtsrm_module.load_state_dict(checkpoint['gtsrm'])
        optimizer.load_state_dict(checkpoint['optimizer'])

        start_epoch = checkpoint['epoch'] + 1
        if 'loss' in checkpoint:
            best_loss = checkpoint['loss']
            best_epoch = checkpoint['epoch']

        scheduler.current_epoch = start_epoch

    for e in range(start_epoch, EPOCHS):
        epoch_start_time = time.time()

        current_lr, stage = scheduler.step()

        torch.cuda.empty_cache()
        gc.collect()

        loss_list = []
        l1_list = []
        ssim_list = []
        grad_list = []
        ssim_raw_list = []
        valid_batches = 0
        skipped_batches = 0
        PRINT_FREQ = 10

        vgg_weight_over_list.clear()
        vgg_weight_under_list.clear()

        pbar = tqdm(
            enumerate(train_loader),
            total=len(train_loader),
            desc=f"Epoch {e + 1}",
            ncols=120,
            bar_format='{l_bar}{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}]'
        )

        for i, imgs in pbar:
            try:
                if imgs is None or imgs[0] is None or imgs[1] is None:
                    continue

                original_img1 = imgs[0].to(device, non_blocking=True)
                original_img6 = imgs[1].to(device, non_blocking=True)

                shared_encoder.train()
                fusion_decoder.train()
                dgcfm_module.train()
                gtsrm_module.train()

                with torch.no_grad():
                    img_enh_over = ahsm_module(original_img1)

                    img_enh_under, _, _, _ = purm_model(original_img6)
                    for b_idx in range(img_enh_under.shape[0]):
                        bias = (torch.randn(img_enh_under[b_idx:b_idx + 1].shape) *
                                (noise_level_img / 255.)).to(device)
                        img_enh_under[b_idx:b_idx + 1] = img_enh_under[b_idx:b_idx + 1] + bias
                        sigma = torch.full((1, 1, 1, 1), noise_level_img / 255.).type_as(
                            img_enh_under[b_idx:b_idx + 1])
                        img_enh_under[b_idx:b_idx + 1] = Denoise_model(
                            img_enh_under[b_idx:b_idx + 1], sigma)

                feat_orig_over = shared_encoder(original_img1)
                feat_orig_under = shared_encoder(original_img6)
                feat_enh_over = shared_encoder(img_enh_over)
                feat_enh_under = shared_encoder(img_enh_under)

                C_fused = dgcfm_module(feat_orig_over, feat_orig_under)
                fused_feat = gtsrm_module(feat_enh_over, feat_enh_under, C_fused)
                fused_img = fusion_decoder(fused_feat)

                with torch.no_grad():
                    weight_list = compute_vgg_weights(
                        feature_model, original_img1, original_img6, temperature=0.01
                    )
                    vgg_weight_over_list.append(weight_list[:, 0].mean().item())
                    vgg_weight_under_list.append(weight_list[:, 1].mean().item())

                loss, L_L1, L_SSIM, L_Grad = Loss_Fun(
                    e, fused_img, original_img1, original_img6, weights=weight_list
                )

                with torch.no_grad():
                    raw_ssim_val = 0.5 * (
                            ms_ssim(original_img1, fused_img, data_range=1) +
                            ms_ssim(original_img6, fused_img, data_range=1)
                    )
                    ssim_raw_list.append(raw_ssim_val.item())

                if torch.isnan(loss) or torch.isinf(loss) or loss.item() > 1000:
                    skipped_batches += 1
                    del loss, L_L1, L_SSIM, L_Grad, raw_ssim_val
                    torch.cuda.empty_cache()
                    continue

                optimizer.zero_grad()
                loss.backward()

                has_nan_inf = False
                for p in trainable_params:
                    if p.grad is not None:
                        if torch.isnan(p.grad).any() or torch.isinf(p.grad).any():
                            has_nan_inf = True
                            break

                if has_nan_inf:
                    optimizer.zero_grad()
                    skipped_batches += 1
                    del loss, L_L1, L_SSIM, L_Grad
                    torch.cuda.empty_cache()
                    continue

                optimizer.step()

                loss_list.append(loss.item())
                l1_list.append(L_L1.item())
                ssim_list.append(L_SSIM.item())
                grad_list.append(L_Grad.item())
                valid_batches += 1

                pbar.set_postfix({
                    'Loss': f'{loss.item():.3f}',
                    'L1': f'{L_L1.item():.3f}',
                    'SSIM': f'{L_SSIM.item():.3f}',
                    'ssim': f'{raw_ssim_val.item():.3f}'
                })

                del loss, L_L1, L_SSIM, L_Grad, raw_ssim_val
                del fused_img, fused_feat, C_fused
                del feat_orig_over, feat_orig_under, feat_enh_over, feat_enh_under
                del img_enh_over, img_enh_under

                if i % 20 == 0 and i > 0:
                    torch.cuda.empty_cache()
                    gc.collect()

            except RuntimeError as ex:
                if "out of memory" in str(ex):
                    torch.cuda.empty_cache()
                    gc.collect()
                optimizer.zero_grad()
                skipped_batches += 1
                continue
            except Exception as ex:
                optimizer.zero_grad()
                skipped_batches += 1
                continue

        pbar.close()
        torch.cuda.empty_cache()
        gc.collect()

        if len(loss_list) > 0:
            epoch_time = time.time() - epoch_start_time
            epoch_times.append(epoch_time)
            avg_loss = np.mean(loss_list)
            avg_l1 = np.mean(l1_list)
            avg_ssim = np.mean(ssim_list)
            avg_grad = np.mean(grad_list)
            avg_raw_ssim = np.mean(ssim_raw_list) if ssim_raw_list else 0
            avg_w_over = np.mean(vgg_weight_over_list) if vgg_weight_over_list else 0
            avg_w_under = np.mean(vgg_weight_under_list) if vgg_weight_under_list else 0
            Loss_list.append(avg_loss)

            state = {
                'shared_encoder': shared_encoder.state_dict(),
                'fusion_decoder': fusion_decoder.state_dict(),
                'dgcfm': dgcfm_module.state_dict(),
                'gtsrm': gtsrm_module.state_dict(),
                'optimizer': optimizer.state_dict(),
                'epoch': e,
                'loss': avg_loss,
            }
            torch.save(state, os.path.join(args.model_path, f'epoch_{e + 1}.pth'))

            if avg_loss < best_loss:
                best_loss = avg_loss
                best_epoch = e
                torch.save(state, os.path.join(args.model_path, 'best_model.pth'))

if __name__ == '__main__':
    main()