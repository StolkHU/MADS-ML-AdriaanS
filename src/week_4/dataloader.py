from mads_datasets import DatasetFactoryProvider, DatasetType
from torchvision import transforms


class FlowerDataLoader:
    def __init__(self, config):
        self.batch_size = config["batch_size"]
        self.augment = config.get("augment", True)
        self.resize_size = config.get("resize_size", 256)
        self.crop_size = config.get("crop_size", 224)
        self.image_size = config.get("image_size", 224)

    def load_data(self):
        # Transformaties op basis van augmentatie
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

        # Dataset ophalen via mads_datasets
        flowers_factory = DatasetFactoryProvider.create_factory(DatasetType.FLOWERS)
        streamers = flowers_factory.create_datastreamer(
            batchsize=self.batch_size,
            train_transform=train_transform,
            val_transform=val_transform,
        )

        return streamers.train, streamers.val
