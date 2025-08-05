import torch
import numpy as np
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
from sklearn.decomposition import PCA
from tqdm import tqdm
import lmdb
from torch.utils.data import Dataset, DataLoader
from io import BytesIO

class LatentDataset(Dataset):
    def __init__(self, lmdb_path):
        self.env = lmdb.open(lmdb_path, readonly=True, lock=False, readahead=False, meminit=False)
        with self.env.begin(write=False) as txn:
            self.length = int(txn.get(b'length').decode('utf-8'))

    def __len__(self):
        return self.length

    def __getitem__(self, idx):
        with self.env.begin(write=False) as txn:
            key = f'{idx:06d}'.encode('ascii')
            latent_bytes = txn.get(key)
        
        buffer = BytesIO(latent_bytes)
        return torch.load(buffer)

def extract_latents_from_lmdb(dataloader, device='cuda', max_batches=50):
    """从 DataLoader 中提取潜在向量"""
    latents = []
    with torch.no_grad():
        for i, batch in enumerate(tqdm(dataloader)):
            if i >= max_batches:
                break
            latents.append(batch.to(device).cpu())
    return torch.cat(latents).numpy()

def visualize_latents(latents, labels=None, method='tsne', title='Latent Space'):
    """使用 PCA 或 t-SNE 可视化潜在向量"""
    if method == 'pca':
        reducer = PCA(n_components=2)
    elif method == 'tsne':
        reducer = TSNE(n_components=2, perplexity=30, learning_rate=200, random_state=42)
    else:
        raise ValueError("method must be 'pca' or 'tsne'")
    
    # Reshape latents if they are not 2D
    if len(latents.shape) > 2:
        latents = latents.reshape(latents.shape[0], -1)

    z_2d = reducer.fit_transform(latents)

    plt.figure(figsize=(8, 6))
    scatter = plt.scatter(z_2d[:, 0], z_2d[:, 1], c=labels, cmap='tab10', s=5) if labels is not None else plt.scatter(z_2d[:, 0], z_2d[:, 1], s=5)
    plt.title(title)
    if labels is not None:
        plt.colorbar(scatter, label="Class")
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(f'{title.replace(" ", "_")}.png')
    plt.show()

# 用法示例
if __name__ == '__main__':
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    # 从 LMDB 加载 latent
    lmdb_path = '../data/latents.lmdb/'  # 使用配置文件中的路径
    dataset = LatentDataset(lmdb_path)
    dataloader = DataLoader(dataset, batch_size=64, shuffle=False)

    # 提取 latent
    # 注意：我们没有标签，所以 labels=None
    latents = extract_latents_from_lmdb(dataloader, device=device, max_batches=512) # Increase max_batches to load more data_processing

    # 可视化 PCA / t-SNE
    visualize_latents(latents, labels=None, method='pca', title='PCA of Latent Space')
    visualize_latents(latents, labels=None, method='tsne', title='t-SNE of Latent Space')