import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image
import os
import matplotlib.pyplot as plt
from tqdm import tqdm

class CelebADataset(Dataset):
    def __init__(self, root_dir, transform=None):
        self.root_dir = root_dir
        self.transform = transform
        self.image_paths = [os.path.join(root_dir, f)
                            for f in os.listdir(root_dir)
                            if f.endswith('.jpg') or f.endswith('.png')]

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        img_path = self.image_paths[idx]
        image = Image.open(img_path).convert('RGB')
        if self.transform:
            image = self.transform(image)
        return image

def denormalize(img_tensor):
    return img_tensor * 0.5 + 0.5

if __name__ == "__main__":
    MAX_IMAGES = 100
    transform = transforms.Compose([
        transforms.CenterCrop(178),
        transforms.Resize(128),
        transforms.ToTensor(),
        transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
    ])

    dataset = CelebADataset(root_dir='img_celeba', transform=transform)
    dataset.image_paths = dataset.image_paths[:MAX_IMAGES]
    dataloader = DataLoader(dataset, batch_size=32, shuffle=True, num_workers=4)

    preprocessed_images = []

    for batch in tqdm(dataloader, desc="Processing first 100 images"):
        preprocessed_images.append(batch)

    all_images = torch.cat(preprocessed_images, dim=0)[:MAX_IMAGES]

    OUTPUT_DIR = "img_process_celeba"
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    torch.save(all_images, os.path.join(OUTPUT_DIR, "celeba_images.pt"))

    print(f"Saved {all_images.shape[0]} preprocessed images ✅")

    # 可选：显示前几张图像
    for i in range(15):
        img = denormalize(all_images[i])
        img = img.permute(1, 2, 0).numpy()
        plt.imshow(img)
        plt.axis('off')
        plt.show()

