# from mads_datasets import DatasetFactoryProvider, DatasetType
# from mltrainer.preprocessors import BasePreprocessor
# from mltrainer import Trainer, TrainerSettings, ReportTypes, metrics

# import torch
# import torch.optim as optim
# from torch import nn

# fashionfactory = DatasetFactoryProvider.create_factory(DatasetType.FASHION)
# preprocessor = BasePreprocessor()
# streamers = fashionfactory.create_datastreamer(batchsize=64, preprocessor=preprocessor)
# train = streamers["train"]
# valid = streamers["valid"]
# trainstreamer = train.stream()
# validstreamer = valid.stream()

# accuracy = metrics.Accuracy()
# loss_fn = torch.nn.CrossEntropyLoss()
# units = [256, 128, 64]
# batchsize = 64


# class NeuralNetwork(nn.Module):
#     def __init__(
#         self, num_classes: int, units1: int, units2: int, units3: int = None
#     ) -> None:
#         super().__init__()
#         self.num_classes = num_classes
#         self.units1 = units1
#         self.units2 = units2
#         self.units3 = units3
#         self.flatten = nn.Flatten()
#         self.linear_relu_stack = nn.Sequential(
#             nn.Linear(28 * 28, units1),
#             nn.BatchNorm1d(units1),
#             nn.ReLU(),
#             nn.Dropout(0.2),
#             nn.Linear(units1, units2),
#             nn.BatchNorm1d(units2),
#             nn.ReLU(),
#             nn.Dropout(0.2),
#             nn.Linear(units2, units3),
#             nn.BatchNorm1d(units3),
#             nn.ReLU(),
#             nn.Dropout(0.2),
#             nn.Linear(units3, num_classes),
#         )

#     def forward(self, x: torch.Tensor) -> torch.Tensor:
#         x = self.flatten(x)
#         logits = self.linear_relu_stack(x)
#         return logits


# settings = TrainerSettings(
#     epochs=20,
#     metrics=[accuracy],
#     logdir="modellogs",
#     train_steps=len(train),
#     valid_steps=len(valid),
#     reporttypes=[ReportTypes.TENSORBOARD, ReportTypes.TOML],
# )

# for unit1 in units:
#     for unit2 in units:
#         if unit2 <= unit1:
#             for unit3 in units:
#                 if unit3 <= unit2:
#                     model = NeuralNetwork(
#                         num_classes=10, units1=unit1, units2=unit2, units3=unit3
#                     )

#                     trainer = Trainer(
#                         model=model,
#                         settings=settings,
#                         loss_fn=loss_fn,
#                         optimizer=optim.Adam,
#                         traindataloader=trainstreamer,
#                         validdataloader=validstreamer,
#                         scheduler=optim.lr_scheduler.ReduceLROnPlateau,
#                     )
#                     trainer.loop()
