from source_1 import source1


import matplotlib.pyplot as plt
import numpy as np
import torchvision

# functions to show an image


def imshow(img):
    img = img / 2 + 0.5     # unnormalize
    npimg = img.numpy()
    # print labels

    plt.imshow(np.transpose(npimg, (1, 2, 0)))
    print(' '.join('%5s' % source1.classes[labels[j]] for j in range(4)))
    plt.show()

# get some random training images
dataiter = iter(source1.trainloader)
images, labels = dataiter.next()

# show images
imshow(torchvision.utils.make_grid(images))
