import os
import random
import lmdb
import matplotlib.pyplot as plt
import torch
from PIL import Image
from torch.utils.data import Dataset
from torchvision import datasets, transforms
from io import BytesIO

transform = transforms.Compose([
    transforms.Resize((512, 512)),
    transforms.ToTensor(),
    transforms.Normalize([0.5]*3, [0.5]*3),  # 归一化到 [-1,1]
])

def get_cifar10_dataset(root: str = "./data", download: bool = True):
    transform = transforms.Compose([
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize((0.5,), (0.5,))
    ])

    train_dataset = datasets.CIFAR10(
        root = root, train = True, download = download, transform = transform
    )
    return train_dataset



class AFHQDataset(Dataset):
    def __init__(self, root_dir, transform=None):
        """
        Args:
            root_dir (str): 数据集的根目录，包含 cat、dog、wild 三个子目录。
            transform (callable, optional): 对图片进行的转换操作。
        """
        self.root_dir = root_dir
        self.transform = transform

        # 类别名称按字典序排列，并分配标签 0,1,2
        self.classes = sorted(entry.name for entry in os.scandir(root_dir) if entry.is_dir())
        self.class_to_idx = {cls_name: idx for idx, cls_name in enumerate(self.classes)}

        # 收集所有图像路径和对应的标签
        self.samples = []
        for cls_name in self.classes:
            cls_path = os.path.join(root_dir, cls_name)
            for fname in os.listdir(cls_path):
                if fname.lower().endswith((".jpg", ".jpeg", ".png", ".bmp")):
                    path = os.path.join(cls_path, fname)
                    label = self.class_to_idx[cls_name]
                    self.samples.append((path, label))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        path, label = self.samples[idx]
        image = Image.open(path).convert('RGB')  # 保证3通道
        if self.transform:
            image = self.transform(image)
        return image, label


class LatentDataset(Dataset):
    def __init__(self, lmdb_path):
        self.lmdb_path = lmdb_path
        self.env = None
        self.txn = None

        # 预先读取长度（只需一次，主进程中读）
        with lmdb.open(self.lmdb_path, readonly = True, lock = False) as env:
            with env.begin(write = False) as txn:
                self.length = int(txn.get(b'length'))

    def __len__(self):
        return self.length

    def __getitem__(self, index):
        if self.env is None:
            self.env = lmdb.open(self.lmdb_path, readonly = True, lock = False, readahead = False, meminit = False)
            self.txn = self.env.begin(write = False)

        key = f"{index:06d}".encode("ascii")
        buf = self.txn.get(key)
        if buf is None:
            raise KeyError(f"Key {key} not found in LMDB")

        latent = torch.load(BytesIO(buf))
        return latent


def show_transformed_images(dataset: Dataset, n: int = 5):
    indices = random.sample(range(len(dataset)), n)
    images = [dataset[i] for i in indices]
    fig, axes = plt.subplots(1, n, figsize = (3 * n, 3))
    if n == 1:
        axes = [axes]
    for img, ax in zip(images, axes):
        img_np = img.permute(1, 2, 0).numpy()
        img_np = (img_np * 0.5) + 0.5
        img_np = img_np.clip(0, 1)

        ax.imshow(img_np)
        ax.axis('off')

    plt.tight_layout()
    plt.show()
