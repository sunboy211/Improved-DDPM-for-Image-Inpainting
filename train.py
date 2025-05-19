import torch
import torch.optim as optim
import torch.nn.functional as F
import math
torch.cuda.empty_cache()
from torch.utils.data import DataLoader
from tqdm import tqdm
from torchmetrics.image.fid import FrechetInceptionDistance
import numpy as np
import torchvision.models as models
import torch.nn as nn
import matplotlib.pyplot as plt
from DDPM_Unet import UNet
from Discriminator import PatchDiscriminator
import os
import lpips

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
batch_size=32
image_size=64
channels=3
epochs=100
T=1000
learning_rate=2e-4

lpips_model = lpips.LPIPS(net='alex').to(device)
lpips_model.eval()

#noise schedule
def linear_beta_schedule(timesteps, beta_start=1e-4, beta_end=0.02):
    return torch.linspace(beta_start, beta_end, timesteps)

def cosine_beta_schedule(timesteps,s = 0.008):
    steps = torch.arange(timesteps + 1) / timesteps
    alphas_cumprod = torch.cos((steps + s) / (1 + s) * math.pi* 0.5) ** 2
    alphas_cumprod = alphas_cumprod / alphas_cumprod[0]
    betas = 1 - (alphas_cumprod[1:] / alphas_cumprod[: -1])
    return torch.clip(betas, 0.0001, 0.9999)

#some necessary parameters in training
betas_known = linear_beta_schedule(timesteps=1000).to(device)
betas_unknown = cosine_beta_schedule(timesteps=1000).to(device)
alphas_known = 1. - betas_known
alphas_cumprod_known = torch.cumprod(alphas_known, dim=0)
alphas_unknown = 1. - betas_unknown
alphas_cumprod_unknown = torch.cumprod(alphas_unknown, dim=0)


def extract(a, t, x_shape):
    #Extract values from 1D tensor `a` at indices `t`, and reshape to match `x_shape` for broadcasting.
    if t.dim() == 0:
        t = t.view(1)
    batch_size = t.shape[0]
    out = a[t]  # advanced indexing
    return out.reshape(batch_size, *((1,) * (len(x_shape) - 1))).to(t.device)


def generate_random_masks(batch_size, height, width, device, mask_type="random"):
    """
    Generate random masks for inpainting
    code modified from https://github.com/WonwoongCho/Generative-Inpainting-pytorch/blob/master/util.py
    mask_type : " random " , " center " , " brush " , or " half "
    """
    masks = torch.zeros((batch_size, 1, height, width), device=device)

    if mask_type == "center":
        # Center rectangular mask
        h_start = height // 4
        h_end = 3 * height // 4
        w_start = width // 4
        w_end = 3 * width // 4
        masks[:, :, h_start: h_end, w_start: w_end] = 1.0

    elif mask_type == "half":
        # Mask half the image ( vertical split )
        masks[:, :, :, width // 2:] = 1.0

    elif mask_type == "brush":
        # Simulate random brush strokes
        for i in range(batch_size):
            num_strokes = np.random.randint(1, 6)
            for _ in range(num_strokes):
                # Random brush width
                brush_width = np.random.randint(2, 12)
                # Random starting point
                y, x = np.random.randint(0, height), np.random.randint(0, width)
                # Random length
                length = np.random.randint(10, max(10, width // 2))
                # Random direction
                angle = np.random.uniform(0, 2 * np.pi)
                dx, dy = np.cos(angle), np.sin(angle)

                # Draw the stroke
                for j in range(length):
                    y_new, x_new = int(y + j * dy), int(x + j * dx)
                    if 0 <= y_new < height and 0 <= x_new < width:
                        y_start = max(0, y_new - brush_width // 2)
                        y_end = min(height, y_new + brush_width // 2 + 1)
                        x_start = max(0, x_new - brush_width // 2)
                        x_end = min(width, x_new + brush_width // 2 + 1)
                        masks[i, :, y_start: y_end, x_start: x_end] = 1.0
    else:
        # Random block masks
        for i in range(batch_size):
            # Number of blocks to mask
            num_blocks = np.random.randint(1, 5)
            for _ in range(num_blocks):
                block_height = np.random.randint(height // 8, height // 2)
                block_width = np.random.randint(width // 8, width // 2)
                block_y = np.random.randint(0, height - block_height)
                block_x = np.random.randint(0, width - block_width)
                masks[i, :, block_y: block_y + block_height, block_x: block_x + block_width] = 1.0
        # Make mask binary
    masks = (masks > 0.5).float()
    return masks

def get_lr_scheduler(optimizer, schedule_type, num_training_steps):
    """
    Create a learning rate scheduler.
    the idea of warmup_cosine refer to: https://github.com/huggingface/transformers/blob/main/src/transformers/optimization.py
    """
    if schedule_type=="constant":
        return torch.optim.lr_scheduler.LambdaLR(optimizer, lambda _: 1.0)
    elif schedule_type=="cosine":
        return torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max = num_training_steps)
    elif schedule_type == "warmup_cosine":
        def lr_lambda(current_step):
            # Warmup for 10% of training
            warmup_steps = int(0.1 * num_training_steps)
            if current_step < warmup_steps:
                return float(current_step) / float(max(1,warmup_steps))
                # Cosine decay after warmup
            progress = float(current_step - warmup_steps) / \
                           float(max(1, num_training_steps -warmup_steps))
            return 0.5 * (1.0 + math.cos(math.pi * progress))
        return torch.optim.lr_scheduler.LambdaLR(optimizer,lr_lambda)

def forward_diffusion_with_adaptive_noise(x0, mask, t,alphas_cumprod_known, alphas_cumprod_unknown):
    """
    Forward diffusion using mask-dependent noise.
    """
    sqrt_alphas_known = torch.sqrt(alphas_cumprod_known[t]).view(-1, 1, 1, 1)
    sqrt_one_minus_alphas_known = torch.sqrt(1. - alphas_cumprod_known[t]).view(-1, 1, 1, 1)

    sqrt_alphas_unknown = torch.sqrt(alphas_cumprod_unknown[t]).view(-1, 1, 1, 1)
    sqrt_one_minus_alphas_unknown = torch.sqrt(1.-alphas_cumprod_unknown[t]).view(-1, 1, 1, 1)

    noise = torch.randn_like(x0)
    x_known = sqrt_alphas_known * x0 + sqrt_one_minus_alphas_known * noise
    x_unknown = sqrt_alphas_unknown * x0 + sqrt_one_minus_alphas_unknown * noise

    x_t = mask * x_known + (1 - mask) * x_unknown
    
    return x_t, noise

model = UNet(in_channels=channels+4 , # +4 for mask and corrupted image
#model_channels=128,
out_channels=channels,
dropout_p=0.1
#time_emb_dim=256,
#block_channels=(128,256,512,512),
#attention_resolutions=(8,4)
).to(device)

optimizer = optim.Adam(model.parameters(), lr=learning_rate)
scheduler= get_lr_scheduler(optimizer,"warmup_cosine",epochs)
discriminator = PatchDiscriminator().to(device)
disc_optimizer = optim.Adam(discriminator.parameters(), lr=learning_rate)

def train(dataloader,
    model,
    optimizer,
    scheduler,
    discriminator=None,
    vgg_extractor=None,
    epochs=100,
    lambda_diff=1.0,
    lambda_vgg=1.0,
    lambda_adv=0.1):

    model.train()
    best_fid = float('inf')

    for epoch in range(epochs):
        epoch_loss = 0.0
        progress_bar = tqdm(dataloader, desc=f"Epoch {epoch + 1}/{epochs}")
        fid_images_real = []
        fid_images_fake = []

        for step,batch in enumerate(progress_bar):
            real_images = batch.to(device)
            #print(real_images.shape)

            noise = torch.randn_like(real_images)
            t = torch.randint(0, T, (real_images.shape[0],), device=device).long()

            mask_type = np.random.choice(["random", "center", "brush", "half"])
            masks = generate_random_masks(real_images.shape[0], image_size, image_size, device, mask_type = mask_type)

            noisy_images, noise = forward_diffusion_with_adaptive_noise(
                real_images, masks, t, alphas_cumprod_known, alphas_cumprod_unknown
            )
            corrupted_images = real_images * (1 - masks)

            model_input = torch.cat([noisy_images, masks, corrupted_images], dim=1)
            total_loss, diff_loss, vgg_loss, adv_loss, x0_pred = compute_total_loss(
                model, discriminator, model_input, real_images, masks, t, noise,alphas_cumprod_known, alphas_cumprod_unknown,lambda_diff=lambda_diff, lambda_vgg=lambda_vgg, lambda_adv=lambda_adv
            )

            loss = total_loss
            optimizer.zero_grad()
            loss.backward()

            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)  # Gradient clipping
            optimizer.step()
            scheduler.step()
            if step % 100 == 0:
                print(f"Step {step}: LR = {scheduler.get_last_lr()[0]:.6f}")
            epoch_loss += loss.item()
            progress_bar.set_postfix(loss=loss.item(), avg_loss=epoch_loss / (step+ 1))
            if step % 500 == 0:
                generate_samples(dataloader.dataset,model,epoch,step)

            fid_images_real.append(real_images.detach().cpu())
            fid_images_fake.append(x0_pred.detach().cpu())

        # FID calculation
        fid = compute_fid_fn(
            real_images=fid_images_real,  # list of tensors
            fake_images=fid_images_fake,  # list of tensors
            device=device,
            batch_size=32  # or adjust for your GPU memory
        )
        # LPIPS计算
        with torch.no_grad():
            lpips_scores = []
            for real_img_batch, fake_img_batch in zip(fid_images_real, fid_images_fake):
                # LPIPS 要求输入在 [-1, 1]
                real_img_batch = real_img_batch * 2 - 1
                fake_img_batch = fake_img_batch * 2 - 1

                score = lpips_model(fake_img_batch.to(device), real_img_batch.to(device))
                lpips_scores.append(score.cpu())

            lpips_score_mean = torch.cat(lpips_scores, dim=0).mean().item()

        if fid < best_fid:
            best_fid = fid
            torch.save(model.state_dict(), f"best_ddpm_model_fid_{fid:.2f}.pt")

        os.makedirs("checkpoints", exist_ok=True)
        torch.save({'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),}, f'checkpoints/ddpm_inpainting_epoch_{epoch}.pt')

        print(f"Epoch {epoch+1}/{epochs} | Loss: {epoch_loss/len(dataloader):.4f} | FID: {fid:.2f} | LPIPS: {lpips_score_mean:.4f}")

        # Visualization of best denoising results
        os.makedirs("samples/best_denoising", exist_ok=True)
        num_visualize = min(4, real_images.shape[0])  # number of images to visualize
        fig, axs = plt.subplots(num_visualize, 3, figsize=(12, 4 * num_visualize))

        for i in range(num_visualize):
            real_img = (real_images[i].detach().cpu().permute(1, 2, 0) * 0.5 + 0.5).clamp(0, 1).numpy()
            noisy_img = (noisy_images[i].detach().cpu().permute(1, 2, 0) * 0.5 + 0.5).clamp(0, 1).numpy()
            x0_img = (x0_pred[i].detach().cpu().permute(1, 2, 0) * 0.5 + 0.5).clamp(0, 1).numpy()

            axs[i, 0].imshow(real_img)
            axs[i, 0].set_title("Original")
            axs[i, 0].axis('off')

            axs[i, 1].imshow(noisy_img)
            axs[i, 1].set_title("Noisy")
            axs[i, 1].axis('off')

            axs[i, 2].imshow(x0_img)
            axs[i, 2].set_title("Denoised x0_pred")
            axs[i, 2].axis('off')

        plt.tight_layout()
        plt.savefig(f"samples/best_denoising/best_epoch_{epoch}_fid_{fid:.2f}.png")
        plt.close()


# Sample generation function
@torch.no_grad()
def generate_samples(dataset,model,epoch,step,num_samples=4):
    model.eval()
    # Sample some images from the dataset
    eval_batch = next(iter(DataLoader(dataset,batch_size=num_samples,shuffle=True)))
    #real_images = eval_batch[0].to(device)
    if isinstance(eval_batch, (tuple, list)):
        real_images = eval_batch[0].to(device)
    else:
        real_images = eval_batch.to(device)
    # Create center mask for all images
    masks=generate_random_masks(num_samples,image_size,image_size,device,mask_type="center")
    # Create corrupted images
    corrupted_images=real_images*(1-masks)

    # Run inpainting
    inpainted_images = sample_from_model(model, corrupted_images, masks, device=torch.device("cuda" if torch.cuda.is_available() else "cpu"))

    # Save visualization
    fig,axs = plt.subplots(num_samples,3,figsize=(12,4*num_samples))
    for i in range(num_samples):
        # Original image
        img_orig = (real_images[i].cpu().permute(1,2,0)*0.5+0.5).numpy().clip(0,1)
        axs[i,0].imshow (img_orig)
        axs[i,0].set_title("Original")
        axs[i,0].axis ('off')

        # Corrupted image
        img_corrupt = (corrupted_images[i].cpu().permute(1,2,0)*0.5+0.5).numpy().clip(0,1)
        axs[i,1].imshow(img_corrupt)
        axs[i,1].set_title("Masked Input")
        axs[i,1].axis('off')

        # Inpainted image
        img_inpaint = (inpainted_images[i].cpu().permute(1,2,0)*0.5+0.5).numpy().clip(0,1)
        axs[i,2].imshow(img_inpaint)
        axs[i,2].set_title ("Inpainted")
        axs[i,2].axis ('off')

    # Save the figure
    os.makedirs ('samples', exist_ok = True)
    plt.tight_layout ()
    plt.savefig (f'samples/inpainting_epoch{epoch}_step{step}.png')
    plt.close()


def compute_fid_fn(real_images, fake_images, device=None, batch_size=32):
    """
    Compute FID between real and fake images using batched updates.
    Args:
        real_images (List[Tensor]): list of (B, C, H, W) real image batches in [0, 1]
        fake_images (List[Tensor]): list of (B, C, H, W) generated image batches in [0, 1]
        device (torch.device): CUDA or CPU
        batch_size (int): max batch size per FID update
    Returns:
        float: FID score
    """
    device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
    fid = FrechetInceptionDistance(feature=2048).to(device)

    def batch_update(images_list, real_flag):
        for batch in images_list:
            batch = (batch * 255).clamp(0, 255).to(torch.uint8).to(device)
            fid.update(batch, real=real_flag)

    batch_update(real_images, real_flag=True)
    batch_update(fake_images, real_flag=False)

    return fid.compute().item()

@torch.no_grad()
def ddim_sample(model, shape, masked_input, mask, num_steps=50, eta=0.0, device=torch.device("cuda" if torch.cuda.is_available() else "cpu"),
                alphas_cumprod_known=alphas_cumprod_known, alphas_cumprod_unknown=alphas_cumprod_unknown):
    # refer to class DDIMSampler in "https://github.com/Alokia/diffusion-DDIM-pytorch/blob/master/utils/engine.py"
    batch_size = shape[0]
    img = torch.randn(shape, device=device)
    time_steps = torch.linspace(T - 1, 0, num_steps, dtype=torch.long, device=device)

    for i in range(num_steps):
        t = time_steps[i].long().repeat(batch_size)
        # 计算 corrupted 区域：被 mask 覆盖的区域
        corrupted = masked_input * mask
        model_input = torch.cat([img, mask, corrupted], dim=1)
        pred_noise = model(model_input, t)
        # 使用 alphas_cumprod_known 和 alphas_cumprod_unknown 提取 alpha 和 sqrt(alpha)
        alpha_known = extract(alphas_cumprod_known, t, img.shape)
        sqrt_alpha_known = torch.sqrt(alpha_known)
        sqrt_one_minus_alpha_known = torch.sqrt(1 - alpha_known)
        alpha_unknown = extract(alphas_cumprod_unknown, t, img.shape)
        sqrt_alpha_unknown = torch.sqrt(alpha_unknown)
        sqrt_one_minus_alpha_unknown = torch.sqrt(1 - alpha_unknown)
        # 计算 x0_pred: 根据已知和未知区域分别使用已知和未知的 alpha
        x0_known = (img - sqrt_one_minus_alpha_known * pred_noise) / sqrt_alpha_known
        x0_unknown = (img - sqrt_one_minus_alpha_unknown * pred_noise) / sqrt_alpha_unknown
        # 根据 mask 将 x0_pred 重新组合
        x0_pred = mask * x0_known + (1 - mask) * x0_unknown

        if i == num_steps - 1:
            img = x0_pred
        else:
            # 计算下一个时间步的 alpha 和 sigma
            alpha_next_known = extract(alphas_cumprod_known, time_steps[i + 1].long(), img.shape)
            alpha_next_unknown = extract(alphas_cumprod_unknown, time_steps[i + 1].long(), img.shape)
            sigma_known = eta * torch.sqrt((1 - alpha_next_known) / (1 - alpha_known)) * torch.sqrt(1 - alpha_known / alpha_next_known)
            sigma_unknown = eta * torch.sqrt((1 - alpha_next_unknown) / (1 - alpha_unknown)) * torch.sqrt(1 - alpha_unknown / alpha_next_unknown)
            # 使用预测的噪声和下一个时间步的 sigma
            noise = torch.randn_like(img) if eta > 0 else 0.0
            # 生成下一个图像
            img = torch.sqrt(alpha_next_known) * x0_known + torch.sqrt(1 - alpha_next_known - sigma_known**2) * pred_noise + sigma_known * noise
            img = img * mask + masked_input * (1 - mask)  # 保持未遮挡区域

    return img


def sample_from_model(model, masked_input, mask, device = torch.device("cuda" if torch.cuda.is_available() else "cpu")):
    shape = masked_input.shape
    return ddim_sample(model, shape, masked_input, mask, alphas_cumprod_known=alphas_cumprod_known,alphas_cumprod_unknown=alphas_cumprod_unknown, num_steps=50, eta=0.0, device=device)


class VGGPerceptualLoss(nn.Module):
    """
    refer to content: https://github.com/jcjohnson/fast-neural-style
    """
    def __init__(self, layers=['relu3_3'], resize=True):
        super().__init__()
        self.vgg = models.vgg16(pretrained=True).features.eval()
        for param in self.vgg.parameters():
            param.requires_grad = False

        self.layer_name_mapping = {
            'relu1_1': 0,
            'relu1_2': 2,
            'relu2_1': 5,
            'relu2_2': 7,
            'relu3_1': 10,
            'relu3_2': 12,
            'relu3_3': 14,
            'relu4_1': 17,
            'relu4_2': 19,
            'relu4_3': 21,
            'relu5_1': 24,
            'relu5_2': 26,
            'relu5_3': 28
        }
        self.selected_layers = [self.layer_name_mapping[layer] for layer in layers]
        self.resize = resize
        self.criterion = nn.L1Loss()

    def normalize_batch(self, batch):
        mean = torch.tensor([0.485, 0.456, 0.406]).to(batch.device)[None, :, None, None]
        std = torch.tensor([0.229, 0.224, 0.225]).to(batch.device)[None, :, None, None]
        return (batch - mean) / std

    def extract_features(self, x):
        features = []
        for i, layer in enumerate(self.vgg):
            x = layer(x)
            if i in self.selected_layers:
                features.append(x)
        return features

    def forward(self, input, target):
        if input.dim() == 3:
            input = input.unsqueeze(0)
        if target.dim() == 3:
            target = target.unsqueeze(0)

        input = self.normalize_batch(input)
        target = self.normalize_batch(target)

        if self.resize:
            input = nn.functional.interpolate(input, size=(224, 224), mode='bilinear', align_corners=False)
            target = nn.functional.interpolate(target, size=(224, 224), mode='bilinear', align_corners=False)

        input_features = self.extract_features(input)
        target_features = self.extract_features(target)

        loss = sum(self.criterion(f_in, f_target) for f_in, f_target in zip(input_features, target_features))
        return loss / len(input_features)

perceptual_loss_fn = VGGPerceptualLoss(layers=['relu2_2', 'relu3_3']).to(device)

def perceptual_loss(fake, real):
    return perceptual_loss_fn(fake, real)

def compute_total_loss(
    model, discriminator, model_input, real_images, masks, t, noise,
    alphas_cumprod_known, alphas_cumprod_unknown,lambda_diff=1.0, lambda_vgg=1.0, lambda_adv=0.1, lambda_mask_l1=1.0):

    # 模型预测 noise
    predicted_noise = model(model_input, t)
    # 根据自适应噪声的重要性加权 MSE 损失（mask 区域更重）
    weighted_loss = (noise - predicted_noise) ** 2
    diffusion_loss = (weighted_loss * (1 + 4 * masks)).mean()

    # x_t 是使用 adaptive schedule 得到的（model_input[:3]）
    x_t = model_input[:, :3, :, :]
    # 用于 mask 区域的反扩散
    sqrt_alpha_known = extract(torch.sqrt(alphas_cumprod_known), t, real_images.shape)
    sqrt_one_minus_alpha_known = extract(torch.sqrt(1 - alphas_cumprod_known), t, real_images.shape)
    # 用于非 mask 区域的反扩散
    sqrt_alpha_unknown = extract(torch.sqrt(alphas_cumprod_unknown), t, real_images.shape)
    sqrt_one_minus_alpha_unknown = extract(torch.sqrt(1 - alphas_cumprod_unknown), t, real_images.shape)
    # 用两个 schedule 分别恢复 x0
    x0_known = (x_t - sqrt_one_minus_alpha_known * predicted_noise) / sqrt_alpha_known
    x0_unknown = (x_t - sqrt_one_minus_alpha_unknown * predicted_noise) / sqrt_alpha_unknown

    # 将预测的 x0 重建出来（已知区域使用 known 的逆扩散）
    x0_pred = masks * x0_known + (1 - masks) * x0_unknown

    # 使用掩码将未被遮挡区域替换为 ground truth
    corrupted_images = model_input[:, 4:, :, :]  # 第5-7通道是 corrupted image
    x0_pred = corrupted_images * (1 - masks) + x0_pred * masks

    assert real_images.shape == x0_pred.shape, f"Shape mismatch: {real_images.shape} vs {x0_pred.shape}"

    # Perceptual loss (VGG)
    vgg_loss = perceptual_loss(x0_pred, real_images)

    # Adversarial loss
    fake_logits = discriminator(x0_pred)
    real_labels = torch.ones_like(fake_logits)
    adv_loss = F.binary_cross_entropy_with_logits(fake_logits, real_labels)

    # 增加 mask 区域 pixel-wise L1 loss
    mask_l1_loss = F.l1_loss(x0_pred * masks, real_images * masks)

    total_loss = (
            lambda_diff * diffusion_loss +
            lambda_vgg * vgg_loss +
            lambda_adv * adv_loss +
            lambda_mask_l1 * mask_l1_loss
    )

    return total_loss, diffusion_loss.item(), vgg_loss.item(), adv_loss.item(), x0_pred


from torchvision import transforms
from data_loading import CelebADataset

def main():
    batch_size = 32
    image_size = 64
    epochs = 100
    timesteps = 1000
    data_dir = 'img_celeba'

    transform = transforms.Compose([
        transforms.CenterCrop(178),
        transforms.Resize((image_size, image_size)),
        transforms.ToTensor(),
        transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
    ])

    dataset = CelebADataset(root_dir=data_dir, transform=transform)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=4)

    model = UNet(
        in_channels=channels+4,
        #model_channels=128,
        out_channels=channels,
        dropout_p=0.1
        #time_emb_dim=256,
        #block_channels=(128, 256, 512, 512),
        #attention_resolutions=(8, 4)
    ).to(device)
    num_training_steps = epochs * len(dataloader)
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    scheduler = get_lr_scheduler(optimizer, "warmup_cosine", num_training_steps)
    discriminator = PatchDiscriminator().to(device)
    disc_optimizer = optim.Adam(discriminator.parameters(), lr=learning_rate)

    train(
        dataloader,
        model,
        optimizer,
        scheduler,
        discriminator=discriminator,
        vgg_extractor=perceptual_loss_fn,
        epochs=epochs,
    )

if __name__ == "__main__":
    main()
