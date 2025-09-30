import numpy as np
import nibabel as nib
import torchio
import random
import torchvision.transforms.functional as F
from torch import tensor
import os


class Transform:
    def __init__(self, images, labels):
        self.images = tensor(images)
        self.labels = tensor(labels)
        self.subject = torchio.Subject(
            mri=torchio.ScalarImage(tensor=images[np.newaxis, ...]),
            label=torchio.LabelMap(tensor=labels[np.newaxis, ...]),
        )

    def __extract_array(self, function):
        results = function(self.subject)
        aug_img = results.mri.data.numpy()[0]
        aug_label = results.label.data.numpy()[0]
        return aug_img, aug_label

    def random_rotate(self, degrees=(0, 90)):
        degree = random.randint(degrees[0], degrees[1])
        rotate_img = F.rotate(
            self.images, angle=degree, interpolation=F.InterpolationMode.BILINEAR
        )
        rotate_label = F.rotate(
            self.labels, angle=degree, interpolation=F.InterpolationMode.NEAREST
        )
        return rotate_img.numpy(), rotate_label.numpy()

    def adjust_brightness(self):
        transform = torchio.RandomGamma(log_gamma=(-0.3, 0.3))
        aug_img, aug_label = self.__extract_array(transform)
        return aug_img, aug_label

    def add_noise(self):
        transform = torchio.RandomNoise(mean=0, std=(0, 0.05))
        aug_img, aug_label = self.__extract_array(transform)
        return aug_img, aug_label

    def elastic_transform(
        self, num_control_points=7, max_displacement=15.0, locked_borders=2
    ):
        transform = torchio.RandomElasticDeformation(
            num_control_points=num_control_points,
            max_displacement=max_displacement,
            locked_borders=locked_borders,
        )
        aug_img, aug_label = self.__extract_array(transform)
        return aug_img, aug_label


class Augment:
    def __init__(self, imgs, labels):
        self.imgs = imgs
        self.labels = labels

    def __call__(self):
        new_imgs = []
        new_labels = []

        for img, label in zip(self.imgs, self.labels):
            transform = Transform(img, label)
            # Adjust brightness of the image
            bri_img, bri_label = transform.adjust_brightness()
            new_imgs.append(bri_img)
            new_labels.append(bri_label)

            # Rotate image for -90 to -45
            rot_img, rot_label = transform.random_rotate(degrees=(-90, -45))
            new_imgs.append(rot_img)
            new_labels.append(rot_label)

            # Rotate image for -45 to 0
            rot_img2, rot_label2 = transform.random_rotate(degrees=(-45, 0))
            new_imgs.append(rot_img2)
            new_labels.append(rot_label2)

            # Rotate image for  degrees for 0 to 45
            rot_img3, rot_label3 = transform.random_rotate(degrees=(0, 45))
            new_imgs.append(rot_img3)
            new_labels.append(rot_label3)

            # Rotate image for 45 to 90
            rot_img4, rot_label4 = transform.random_rotate(degrees=(45, 90))
            new_imgs.append(rot_img4)
            new_labels.append(rot_label4)

            # Add noise
            noise_img, noise_label = transform.add_noise()
            new_imgs.append(noise_img)
            new_labels.append(noise_label)

            # Elastic transformation num_control_points=7, max_displacement=15.0, locked_borders=2
            elas_img, elas_label = transform.elastic_transform()
            new_imgs.append(elas_img)
            new_labels.append(elas_label)

            # Elastic transformation num_control_points=5, max_displacement=10.0, locked_borders=2
            elas_img2, elas_label2 = transform.elastic_transform(
                num_control_points=5, max_displacement=10.0, locked_borders=2
            )
            new_imgs.append(elas_img2)
            new_labels.append(elas_label2)

            # Elastic transformation num_control_points=7, max_displacement=7, locked_borders=2
            elas_img3, elas_label3 = transform.elastic_transform(
                num_control_points=7, max_displacement=7.0, locked_borders=2
            )
            new_imgs.append(elas_img3)
            new_labels.append(elas_label3)

            # Elastic transformation num_control_points=7, max_displacement=10, locked_borders=2
            elas_img4, elas_label4 = transform.elastic_transform(
                num_control_points=7, max_displacement=7.0, locked_borders=2
            )
            new_imgs.append(elas_img4)
            new_labels.append(elas_label4)

        self.new_imgs = np.stack(new_imgs, axis=0)
        self.new_labels = np.stack(new_labels, axis=0)
        return self.new_imgs, self.new_labels
