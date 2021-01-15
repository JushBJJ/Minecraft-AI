import torch
import torch.nn as nn
import pyscreeze as ps
import numpy as np
import matplotlib.pyplot as plt
import math


class CNN(nn.Module):
    def __init__(self):
        super(CNN, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(1, 32, 5),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(5),
            nn.Conv2d(32, 64, 3),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 64, 3),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(3)
        )

        self.fc = nn.Sequential(
            nn.Linear(279104, 64),
            nn.ReLU(inplace=True),
            nn.Linear(64, 32),
            nn.ReLU(inplace=True),
            nn.Linear(32, 16),
            nn.ReLU(inplace=True),
            nn.Linear(16, 8),
            nn.Tanh()
        )

    def forward(self, x):
        print(x.shape)
        y = self.conv(x)
        print("Yshape: ", y.shape)

        showConvolutions(y.shape[1], y)
        y = y.view(y.size(0), -1)
        print(y.shape)
        y = self.fc(y)

        print(y)
        return y


def showConvolutions(channels, convolution):
    y = convolution[0].detach().numpy()
    cr = int(math.sqrt(channels))
    cols = cr
    rows = cr

    fig = plt.figure()

    x = 0
    for i in range(1, cols*rows+1):
        img = y[x].transpose()
        fig.add_subplot(rows, cols, i)
        plt.imshow(img)
        x += 1

        if x >= channels:
            break

    plt.subplots_adjust()
    plt.show()


def getScreen():
    imageGrayscale = (np.array(ps.screenshot().convert("L").getdata())-128)/128
    image = imageGrayscale.reshape(768, 1366)

    # Convert to 3d
    image = image.reshape(1, image.shape[0], image.shape[1])

    # Convert to 4d
    image = image.reshape(1, image.shape[0], image.shape[1], image.shape[2])

    image = torch.from_numpy(image)  # Convert into torch tensor.
    return image.to(dtype=torch.float32)


model = CNN()
x = getScreen()

model(x)
