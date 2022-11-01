import torch
import torch.nn as nn
import torch.nn.functional as F
from collections import OrderedDict
from tensorflow import keras

model = keras.models.Sequential([
    keras.Input((227, 227, 3)),
    keras.layers.Conv2D(filters=96, kernel_size=(11,11), strides=(4,4), activation='relu'),
    keras.layers.BatchNormalization(),
    keras.layers.MaxPool2D(pool_size=(3,3), strides=(2,2)),
    keras.layers.Conv2D(filters=256, kernel_size=(5,5), strides=(1,1), activation='relu', padding="same"),
    keras.layers.BatchNormalization(),
    keras.layers.Conv2D(filters=384, kernel_size=(3,3), strides=(1,1), activation='relu', padding="same"),
    keras.layers.BatchNormalization(),
    keras.layers.Conv2D(filters=384, kernel_size=(3,3), strides=(1,1), activation='relu', padding="same"),
    keras.layers.BatchNormalization(),
    keras.layers.MaxPool2D(pool_size=(3,3), strides=(2,2)),
    keras.layers.Flatten(),
    keras.layers.Dense(254, activation='relu'),
    keras.layers.Dropout(0.5),
    keras.layers.Dense(10, activation='softmax')
])
model.summary()

'''
# Rewrite this class with AlexNet equivalent
class ANet(nn.Module):
    
    model = nn.Sequential(OrderedDict([
                # Conv 1 55x55x96
                nn.Conv2d(3, 96, kernel_size=11, stride=4),
                nn.BatchNorm2d(96),
                nn.ReLU(),
                # Pool 1 27x27x96
                nn.MaxPool2d(kernel_size=3, stride=2),
                # Conv 2 27x27x256
                nn.Conv2d(3, 256, kernel_size=5, stride=1, padding=0),
                nn.BatchNorm2d(96),
                nn.ReLU(),
                # Conv 3 27x27x384
                nn.Conv2d(3, 384, kernel_size=3, stride=1, padding=0),
                nn.BatchNorm2d(96),
                nn.ReLU(),
                # Conv 4 27x27x384
                nn.Conv2d(3, 384, kernel_size=3, stride=1, padding=0),
                nn.BatchNorm2d(96),
                nn.ReLU(),
                # Pool 2 13x13x384
                nn.MaxPool2d(kernel_size=3, stride=2),
                # Flatten 3 dim to 1 dim
                nn.Flatten(3,1),
                # Dropout 1
                nn.Dropout(0.5),
                # Dense layer 1 size 254
                nn.Linear(64896, 254),
                nn.ReLU(),
                # Dropout 2
                nn.Dropout(0.5),
                # Dense layer 2 size 10
                nn.Linear(254, 10)
            ]))
'''
# Rewrite this class with AlexNet equivalent
class ANet(nn.Module):
    
    def __init__(self, in_dim, out_dim):
        super().__init__()

        self.in_dim = in_dim
        self.out_dim = out_dim

        # Conv 1 55x55x96
        self.conv1 = nn.Sequential(
                        nn.Conv2d(3, 96, kernel_size=11, stride=4, padding=0),
                        nn.BatchNorm2d(96),
                        nn.ReLU()
                        )

        # Pool 1 27x27x96
        self.pool1 = nn.MaxPool2d(kernel_size=3, stride=2)

        # Conv 2 27x27x256
        self.conv2 = nn.Sequential(
                        nn.Conv2d(3, 256, kernel_size=5, stride=1, padding=0),
                        nn.BatchNorm2d(96),
                        nn.ReLU()
                        )

        # Conv 3 27x27x384
        self.conv3 = nn.Sequential(
                        nn.Conv2d(3, 384, kernel_size=3, stride=1, padding=0),
                        nn.BatchNorm2d(96),
                        nn.ReLU()
                        )

        # Conv 4 27x27x384
        self.conv4 = nn.Sequential(
                        nn.Conv2d(3, 384, kernel_size=3, stride=1, padding=0),
                        nn.BatchNorm2d(96),
                        nn.ReLU()
                        )

        # Pool 2 13x13x384
        self.pool2 = nn.MaxPool2d(kernel_size=3, stride=2)

        # Dense layer 1 size 254
        self.lin1 = nn.Sequential(
                        nn.Dropout(0.5),
                        nn.Linear(13*13*384, 254),
                        nn.ReLU()
                        )

        # Dense layer 2 size 10
        self.lin2 = nn.Sequential(
                        nn.Dropout(0.5),
                        nn.Linear(254, self.out_dim)
                        )

    def forward(self, x):
        x = self.conv1(x)
        x = self.pool1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        x = self.conv4(x)
        x = self.pool2(x)

        # flatten convolutional layer into vector
        x = x.view(x.size(0), -1)
        x = self.lin1(x)
        x = self.lin2(x)

        return x

