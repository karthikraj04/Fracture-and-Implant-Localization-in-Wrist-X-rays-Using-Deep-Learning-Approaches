import os
import csv
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader

from model import UNetGenerator, PatchGANDiscriminator

from dataset import MURADataset
import torchvision.utils as vutils
from torchvision import transforms
from torchvision.models import vgg16, VGG16_Weights
import math
import numpy as np
from skimage.metrics import structural_similarity as compare_ssim

# CONFIG
epochs = 120
start_epoch = 106  # Start from your last checkpoint
batch_size = 8
fine_tune_lr = 1e-5  # Lowered LR

save_path = "/content/drive/MyDrive/fracture-gan/saved_models"
sample_path = "/content/drive/MyDrive/fracture-gan/samples"
log_path = "/content/drive/MyDrive/fracture-gan/logs"
data_dir = "/content/drive/MyDrive/fracture-gan/data/processed/negative"
use_perceptual_loss = True

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("✅ Using", device)

os.makedirs(save_path, exist_ok=True)
os.makedirs(sample_path, exist_ok=True)
os.makedirs(log_path, exist_ok=True)

# Logging CSV file
csv_file = os.path.join(log_path, "fine_tune_log.csv")
if not os.path.exists(csv_file):
    with open(csv_file, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["epoch", "batch", "g_loss", "d_loss", "psnr", "ssim"])

# DATASET
dataset = MURADataset(root_dir=data_dir, sliding_mask=True)
dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=0)
print(f"[INFO] Dataset loaded with {len(dataset)} samples")

# MODELS
generator = UNetGenerator().to(device)
discriminator = PatchGANDiscriminator().to(device)

# LOAD CHECKPOINTS
gen_ckpt = os.path.join(save_path, f"generator_epoch{start_epoch}.pth")
disc_ckpt = os.path.join(save_path, f"discriminator_epoch{start_epoch}.pth")

if os.path.exists(gen_ckpt):
    generator.load_state_dict(torch.load(gen_ckpt))
    print(f"🔁 Generator resumed from epoch {start_epoch}")
if os.path.exists(disc_ckpt):
    discriminator.load_state_dict(torch.load(disc_ckpt))
    print(f"🔁 Discriminator resumed from epoch {start_epoch}")

# LOSSES
reconstruction_loss = nn.L1Loss()
adv_loss_fn = nn.BCEWithLogitsLoss()

if use_perceptual_loss:
    vgg = vgg16(weights=VGG16_Weights.IMAGENET1K_V1).features[:16].to(device).eval()
    for p in vgg.parameters():
        p.requires_grad = False

    normalize = transforms.Normalize([0.485, 0.456, 0.406],
                                     [0.229, 0.224, 0.225])

    def perceptual_loss(fake, real):
        fake_rgb = normalize(fake.repeat(1, 3, 1, 1))
        real_rgb = normalize(real.repeat(1, 3, 1, 1))
        return reconstruction_loss(vgg(fake_rgb), vgg(real_rgb))

# OPTIMIZERS
optimizer_G = optim.Adam(generator.parameters(), lr=fine_tune_lr, betas=(0.5, 0.999))
optimizer_D = optim.Adam(discriminator.parameters(), lr=fine_tune_lr, betas=(0.5, 0.999))

def psnr_metric(img1, img2):
    mse = torch.mean((img1 - img2) ** 2).item()
    if mse == 0:
        return 100
    PIXEL_MAX = 1.0  # Assuming images normalized between 0 and 1
    return 20 * math.log10(PIXEL_MAX / math.sqrt(mse))

def ssim_metric(img1, img2):
    img1_np = img1.detach().squeeze().cpu().numpy()
    img2_np = img2.detach().squeeze().cpu().numpy()
    if img1_np.ndim == 3:  # batch dimension
        ssim_vals = []
        for i in range(img1_np.shape[0]):
            ssim_vals.append(compare_ssim(img1_np[i], img2_np[i], data_range=img2_np[i].max() - img2_np[i].min()))
        return np.mean(ssim_vals)
    else:
        return compare_ssim(img1_np, img2_np, data_range=img2_np.max() - img2_np.min())

# TRAINING LOOP
for epoch in range(start_epoch + 1, epochs + 1):
    print(f"\n🔧 Fine-Tune Epoch {epoch}")

    generator.train()
    discriminator.train()

    for i, (masked_imgs, real_imgs) in enumerate(dataloader):
        masked_imgs, real_imgs = masked_imgs.to(device), real_imgs.to(device)

        # ------------------
        #  Train Generator
        # ------------------
        optimizer_G.zero_grad()
        fake_imgs = generator(masked_imgs)

        # Adversarial loss (try to fool discriminator)
        pred_fake = discriminator(fake_imgs, masked_imgs)
        valid = torch.ones_like(pred_fake, device=device)
        g_adv_loss = adv_loss_fn(pred_fake, valid)

        # Reconstruction loss
        rec_loss = reconstruction_loss(fake_imgs, real_imgs)

        # Perceptual loss
        if use_perceptual_loss:
            perc_loss = perceptual_loss(fake_imgs, real_imgs)
            g_loss = rec_loss + 0.1 * perc_loss + 0.001 * g_adv_loss
        else:
            g_loss = rec_loss + 0.001 * g_adv_loss

        g_loss.backward()
        optimizer_G.step()

        # ---------------------
        #  Train Discriminator
        # ---------------------
        optimizer_D.zero_grad()
        pred_real = discriminator(real_imgs, masked_imgs)
        valid = torch.ones_like(pred_real, device=device)
        loss_real = adv_loss_fn(pred_real, valid)

        pred_fake_detach = discriminator(fake_imgs.detach(), masked_imgs)
        fake = torch.zeros_like(pred_fake_detach, device=device)
        loss_fake = adv_loss_fn(pred_fake_detach, fake)

        d_loss = 0.5 * (loss_real + loss_fake)
        d_loss.backward()
        optimizer_D.step()

        # Normalize images to [0,1] for metrics (assumes inputs are in range [-1,1])
        fake_norm = (fake_imgs + 1) / 2
        real_norm = (real_imgs + 1) / 2

        batch_psnr = psnr_metric(fake_norm, real_norm)
        batch_ssim = ssim_metric(fake_norm, real_norm)

        if i % 100 == 0:
            print(f"[Epoch {epoch}/{epochs}] [Batch {i}/{len(dataloader)}] "
                  f"G loss: {g_loss.item():.4f} D loss: {d_loss.item():.4f} PSNR: {batch_psnr:.4f} SSIM: {batch_ssim:.4f}")

        # Log every batch
        with open(csv_file, "a", newline="") as f:
            writer = csv.writer(f)
            writer.writerow([epoch, i, f"{g_loss.item():.6f}", f"{d_loss.item():.6f}", f"{batch_psnr:.6f}", f"{batch_ssim:.6f}"])

    # SAVE MODELS
    torch.save(generator.state_dict(), f"{save_path}/generator_epoch{epoch}.pth")
    torch.save(discriminator.state_dict(), f"{save_path}/discriminator_epoch{epoch}.pth")

    # SAVE SAMPLES
    generator.eval()
    with torch.no_grad():
        sample_input, sample_gt = next(iter(dataloader))
        sample_input = sample_input.to(device)
        sample_fake = generator(sample_input)
        combined = torch.cat((sample_input.cpu(), sample_fake.cpu(), sample_gt.cpu()), dim=0)
        vutils.save_image(combined, f"{sample_path}/combined_epoch{epoch}.png",
                          nrow=sample_input.size(0), normalize=True)

    print(f"✅ Epoch {epoch} complete.\n")

print("🎉 Fine-tuning complete.")
