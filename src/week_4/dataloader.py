from mads_datasets import DatasetFactoryProvider, DatasetType
from torchvision import transforms
import torch


class AugmentPreprocessor:
    """Preprocessor voor batch augmentatie"""

    def __init__(self, transform):
        self.transform = transform

    def __call__(self, batch):
        X, y = zip(*batch)
        X = [self.transform(x) for x in X]
        return torch.stack(X), torch.stack(y)


class FlowerDataLoader:
    """FlowerDataLoader"""

    def __init__(self, config):
        self.batch_size = config["batch_size"]
        self.augment = config.get("augment", True)
        self.resize_size = config.get("resize_size", 256)
        self.crop_size = config.get("crop_size", 224)
        self.image_size = config.get("image_size", 224)

    def load_data(self):
        """Laad de data in"""

        # Maak factory
        flowers_factory = DatasetFactoryProvider.create_factory(DatasetType.FLOWERS)

        # Maak streamers
        streamers = flowers_factory.create_datastreamer(batchsize=self.batch_size)

        # Definieer transforms (ZONDER ToTensor - data is al tensor!)
        if self.augment:
            train_transform = transforms.Compose(
                [
                    transforms.ToPILImage(),  # Eerst naar PIL voor transforms
                    transforms.RandomResizedCrop(self.crop_size),
                    transforms.RandomHorizontalFlip(),
                    transforms.ToTensor(),  # Dan weer naar tensor
                    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
                ]
            )
        else:
            train_transform = transforms.Compose(
                [
                    transforms.ToPILImage(),  # Eerst naar PIL voor transforms
                    transforms.Resize((self.image_size, self.image_size)),
                    transforms.ToTensor(),  # Dan weer naar tensor
                    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
                ]
            )

        val_transform = transforms.Compose(
            [
                transforms.ToPILImage(),  # Eerst naar PIL voor transforms
                transforms.Resize(self.resize_size),
                transforms.CenterCrop(self.crop_size),
                transforms.ToTensor(),  # Dan weer naar tensor
                transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
            ]
        )

        # Maak preprocessors
        train_preprocessor = AugmentPreprocessor(train_transform)
        valid_preprocessor = AugmentPreprocessor(val_transform)

        # Krijg de streamers
        train_streamer = streamers["train"]
        valid_streamer = streamers["valid"]

        # Stel preprocessors in (zoals in jouw voorbeeld)
        train_streamer.preprocessor = train_preprocessor
        valid_streamer.preprocessor = valid_preprocessor

        return train_streamer, valid_streamer
