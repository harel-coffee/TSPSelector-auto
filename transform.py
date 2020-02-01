import numpy as np
from torchvision import  transforms
from math import pi, sin, cos
import torch


class RandomRotate(object):
    """
    rotate the original coordinates by an random  angle
    """
    def __init__(self, num_rotate = 8):
        self.rotate_time = num_rotate

    def __call__(self, x):
        new_x = np.zeros(shape=x.shape)
        rotate_idx = np.random.randint(self.rotate_time)
        angle = 2 * pi * float(rotate_idx) / self.rotate_time
        new_x[:, 0] = x[:, 0] * cos(angle) - x[:, 1] * sin(angle)
        new_x[:, 1] = x[:, 0] * sin(angle) + x[:, 1] * cos(angle)
        return new_x


class Normalize(object):
    """
    normalize the unnormalized coordinates
    """
    def __init__(self):
        pass
    def __call__(self, x):
        x_min, x_max = x[:, 0].min(), x[:, 0].max()
        y_min, y_max = x[:, 1].min(), x[:, 1].max()
        x[:, 0], x[:, 1] = x[:, 0] - x_min, x[:, 1] - y_min
        scale = max(x_max - x_min, y_max - y_min)
        x = x / scale
        return x


class ToImage(object):
    """
    coordinates to image
    """
    def __init__(self, num_grid = 64):
        self.num_grid = num_grid
    def __call__(self, x):
        image = np.zeros((self.num_grid, self.num_grid), dtype=np.float32)
        for i in range(x.shape[0]):
            idx_x = int(x[i][0] * self.num_grid) - 1
            idx_y = int(x[i][1] * self.num_grid) - 1
            image[idx_x][idx_y] = image[idx_x][idx_y] + 1.0
        return image


class RandomFlip(object):
    """
    Random flip the image
    """
    def __init__(self):
        pass
    def __call__(self, image):
        axis = np.random.randint(3)
        if axis == 2:
            return image
        image = np.flip(image, axis=axis)
        return image

class CovertToTensor(object):
    """
    convert image to Tensor
    """
    def __init__(self):
        pass
    def __call__(self, image):
        image = np.ascontiguousarray(image, dtype=np.float32)
        image = torch.from_numpy(image)
        return image


class Interpolate(object):
    """
    reduce the image resolution by scale factor
    """
    def __init__(self, scale_factor):
        self.scale_factor = scale_factor
    def __call__(self, image):
        init_size = image.shape[0]
        image = image.view((1, 1, init_size, init_size))
        image = torch.nn.functional.interpolate(image, scale_factor = self.scale_factor)
        image = image.view((init_size * self.scale_factor, init_size * self.scale_factor))
        return image

"""
default coordinates to image transformations
image size (num_gird * scale_factor, num_grid * scale_factor)
"""
default_train_transforms = transforms.Compose([
    RandomRotate(num_rotate = 8),
    Normalize(),
    ToImage(num_grid= 64),
    RandomFlip(),
    CovertToTensor(),
    Interpolate(scale_factor= 4)
])

default_val_transforms = transforms.Compose([
    Normalize(),
    ToImage(num_grid= 64),
    CovertToTensor(),
    Interpolate(scale_factor= 4)
])



class BuildTransformation(object):
    def __init__(self, num_rotate, num_grid, scale_factor, flip = True):
        self.random_rotate = RandomRotate(num_rotate = num_rotate) if num_rotate > 0 else None
        self.normalize = Normalize()
        self.to_image = ToImage(num_grid= num_grid)
        self.random_flip = RandomFlip() if flip else None
        self.convert_to_tensor = CovertToTensor()
        self.interpolate = Interpolate(scale_factor= scale_factor) if scale_factor > 1 else None

    def get_train_transform(self):
        trans = []
        if self.random_rotate:
            trans.append(self.random_rotate)
        trans.append(self.normalize)
        trans.append(self.to_image)
        if self.random_flip:
            trans.append(self.random_flip)
        trans.append(self.convert_to_tensor)
        if self.interpolate:
            trans.append(self.interpolate)
        return transforms.Compose(trans)

    def get_val_transform(self):
        trans = []
        trans.append(self.normalize)
        trans.append(self.to_image)
        trans.append(self.convert_to_tensor)
        if self.interpolate:
            trans.append(self.interpolate)
        return transforms.Compose(trans)


