from torchvision import transforms
from torch.utils.data import DataLoader
from torchvision import datasets


class FlowerDataLoader:
    def __init__(
        self,
        batch_size=32,
        image_size=224,
        augment=True,
        resize_size=256,
        crop_size=224,
    ):
        self.batch_size = batch_size
        self.image_size = image_size
        self.augment = augment
        self.resize_size = resize_size
        self.crop_size = crop_size

    def load_data(self):
        if self.augment:
            train_transform = transforms.Compose(
                [
                    transforms.RandomResizedCrop(self.crop_size),
                    transforms.RandomHorizontalFlip(),
                    transforms.ToTensor(),
                    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
                ]
            )
        else:
            train_transform = transforms.Compose(
                [
                    transforms.Resize((self.image_size, self.image_size)),
                    transforms.ToTensor(),
                    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
                ]
            )

        val_transform = transforms.Compose(
            [
                transforms.Resize(self.resize_size),
                transforms.CenterCrop(self.crop_size),
                transforms.ToTensor(),
                transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
            ]
        )

        train_data = datasets.Flowers102(
            root="./data", split="train", download=True, transform=train_transform
        )
        val_data = datasets.Flowers102(
            root="./data", split="val", download=True, transform=val_transform
        )

        train_loader = DataLoader(train_data, batch_size=self.batch_size, shuffle=True)
        val_loader = DataLoader(val_data, batch_size=self.batch_size, shuffle=False)

        return train_loader, val_loader
