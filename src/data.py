from pathlib import Path

import lightning
import pandas as pd
import timm
import torch
from PIL import Image
from torch.utils.data import DataLoader, Dataset, TensorDataset
from torchvision import transforms

from src.models import build_network


def build_transform(img_size, mean, std):
    return transforms.Compose([
        transforms.Resize((img_size, img_size)),
        transforms.ToTensor(),
        transforms.Normalize(mean, std),
    ])


class ImageDataset(Dataset):
    def __init__(self, root, paths, labels, transform):
        self.root = Path(root)
        self.paths = paths
        self.labels = labels
        self.transform = transform

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, i):
        with Image.open(self.root / self.paths[i]) as img:
            img = img.convert("RGB")
        return self.transform(img), self.labels[i]


class ArtworkDataModule(lightning.LightningDataModule):
    """Reads the splits csv (one row per image: path, label, split) and builds the dataloaders."""
    def __init__(
        self,
        splits: str,
        data_root: str,
        model_name: str,
        img_size: int = 224,
        batch_size: int = 64,
        num_workers: int = 4,
        path_col: str = "images_path",
        label_col: str = "label",
        split_col: str = "split",
        cache_features: bool = False,
        features_dir: str = "features",
    ) -> None:
        super().__init__()
        self.save_hyperparameters()
        self.data_root = data_root
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.model_name = model_name
        self.img_size = img_size
        self.cache_features = cache_features
        self.features_dir = Path(features_dir)
        self.features = None

        df = pd.read_csv(splits)
        self.classes = sorted(df[label_col].unique())
        class_id = {c: i for i, c in enumerate(self.classes)}

        self.paths, self.labels = {}, {}
        for split in ["train", "val", "test"]:
            rows = df[df[split_col] == split]
            self.paths[split] = rows[path_col].tolist()
            self.labels[split] = rows[label_col].map(class_id).tolist()

        cfg = timm.get_pretrained_cfg(model_name)
        self.transform = build_transform(img_size, cfg.mean, cfg.std)

    def setup(self, stage):
        if self.cache_features and self.features is None:
            self.features = self.load_features()

    def load_features(self):
        path = self.features_dir / f"{self.model_name}_{self.img_size}.pt"
        if path.exists():
            cached = torch.load(path)
            if cached["paths"] == self.paths:
                print(f"using cached features {path}")
                return {split: f.float() for split, f in cached["features"].items()}
            print(f"{path} was made from other splits, extracting again")

        device = "cuda" if torch.cuda.is_available() else "cpu"
        backbone = build_network(self.model_name, 0, self.img_size, pretrained=True).to(device).eval()  # 0 = no head
        splits = ["train", "val", "test"]
        paths = sum((self.paths[s] for s in splits), [])
        labels = sum((self.labels[s] for s in splits), [])
        print(f"extracting features of {len(paths)} images with {self.model_name}")
        loader = DataLoader(ImageDataset(self.data_root, paths, labels, self.transform),
                            batch_size=self.batch_size, num_workers=self.num_workers)
        with torch.no_grad():
            feats = torch.cat([backbone(x.to(device)).cpu() for x, _ in loader])
        features = dict(zip(splits, torch.split(feats, [len(self.paths[s]) for s in splits])))

        self.features_dir.mkdir(parents=True, exist_ok=True)
        # saved in fp16 to halve the size on disk
        torch.save({"paths": self.paths, "features": {s: f.half() for s, f in features.items()}}, path)
        print(f"saved {path}")
        return features

    def _loader(self, split):
        train = split == "train"
        if self.cache_features:
            dataset = TensorDataset(self.features[split], torch.tensor(self.labels[split]))
            workers = 0  # already in memory
        else:
            dataset = ImageDataset(self.data_root, self.paths[split], self.labels[split], self.transform)
            workers = self.num_workers
        return DataLoader(dataset, batch_size=self.batch_size, shuffle=train, drop_last=train,
                          num_workers=workers, pin_memory=True, persistent_workers=workers > 0)

    def train_dataloader(self):
        return self._loader("train")

    def val_dataloader(self):
        return self._loader("val")

    def test_dataloader(self):
        return self._loader("test")