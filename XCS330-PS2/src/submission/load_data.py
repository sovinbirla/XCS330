import numpy as np
import os
import random
import torch
from torch.utils.data import IterableDataset
import time
import imageio


def get_images(paths, labels, nb_samples=None):
    """
    Takes a set of character folders and labels and returns paths to image files
    paired with labels.
    Args:
        paths: A list of character folders
        labels: List or numpy array of same length as paths
        nb_samples: Number of images to retrieve per character
    Returns:
        image_path = characters
        List of (label, image_path) tuples
    """
    if nb_samples is not None:
        sampler = lambda x: random.sample(x, nb_samples)
    else:
        sampler = lambda x: x
    image_labels = [
        (i, os.path.join(path, image))
        for i, path in zip(labels, paths)
        for image in sampler(os.listdir(path))
    ]

    return image_labels


class DataGenerator(IterableDataset):
    """
    Data Generator capable of generating batches of Omniglot data.
    A "class" is considered a class of omniglot digits.
    """

    def __init__(
        self,
        num_classes,
        num_samples_per_class,
        batch_type,
        config={},
        cache=True,
    ):
        """
        Args:
            num_classes: Number of classes for classification (N-way)
            num_samples_per_class: num samples to generate per class in one batch (K+1)
            batch_type: train/val/test
            config: data_folder - folder where the data is located
                    img_size - size of the input images
            cache: whether to cache the images loaded
        """
        self.num_samples_per_class = num_samples_per_class # K+1
        self.num_classes = num_classes # N

        data_folder = config.get("data_folder", "./omniglot_resized")
        self.img_size = config.get("img_size", (28, 28))

        self.dim_input = np.prod(self.img_size)
        self.dim_output = self.num_classes

        character_folders = [
            os.path.join(data_folder, family, character)
            for family in os.listdir(data_folder)
            if os.path.isdir(os.path.join(data_folder, family))
            for character in os.listdir(os.path.join(data_folder, family))
            if os.path.isdir(os.path.join(data_folder, family, character))
        ]

        random.seed(1)
        random.shuffle(character_folders)
        num_val = 100
        num_train = 1100
        self.metatrain_character_folders = character_folders[:num_train]
        self.metaval_character_folders = character_folders[num_train : num_train + num_val]
        self.metatest_character_folders = character_folders[num_train + num_val :]
        self.image_caching = cache
        self.stored_images = {}

        if batch_type == "train":
            self.folders = self.metatrain_character_folders
        elif batch_type == "val":
            self.folders = self.metaval_character_folders
        else:
            self.folders = self.metatest_character_folders

        self.sample_fn = random.sample

        self.shuffle_fn = np.random.shuffle

    def image_file_to_array(self, filename, dim_input):
        """
        Takes an image path and returns numpy array
        Args:
            filename: Image filename
            dim_input: Flattened shape of image
        Returns:
            1 channel image
        """
        if self.image_caching and (filename in self.stored_images):
            return self.stored_images[filename]
        image = imageio.imread(filename)  # misc.imread(filename)
        image = image.reshape([dim_input])
        image = image.astype(np.float32) / image.max()
        image = 1.0 - image
        if self.image_caching:
            self.stored_images[filename] = image
        return image
    
    def _sample(self, sample_fn=None, shuffle_fn=None):
        """
        Samples a batch for training, validation, or testing
        Args:
            sample_fn - pointer to a sampling function to be used
            shuffle_fn - pointer to a shuffling function to be used
        Returns:
            A tuple of (1) Image batch and (2) Label batch:
                1. image batch has shape [K+1, N, 784] and is a numpy array
                2. label batch has shape [K+1, N, N] and is a numpy array
            where K is the number of "shots", N is the number of classes
        Note:
            1. The numpy functions np.random.shuffle and np.eye (for creating)
            one-hot vectors would be useful.

            2. For shuffling, remember to make sure images and labels are shuffled
            in the same order, otherwise the one-to-one mapping between images
            and labels may get messed up. Hint: there is a clever way to use
            np.random.shuffle here.
            
            3. The value for `self.num_samples_per_class` will be set to K+1 
            since for K-shot classification you need to sample K supports and 
            1 query.

            4. PyTorch uses float32 as the default for representing model parameters. 
            You would need to return numpy arrays with the same datatype
        """
        if sample_fn is None:
            sample_fn = self.sample_fn
        if shuffle_fn is None:
            shuffle_fn = self.shuffle_fn

        N, K = self.num_classes, self.num_samples_per_class
        diff_characters = sample_fn(self.folders, N)
        # print("diff_characters: ", diff_characters)
        image_labels = np.eye(N, dtype=np.float32) 

        characters = get_images(diff_characters, image_labels, nb_samples=K)
        support_images, support_labels = [], []
        query_images, query_labels = [], []

        # print("K:" , K) 
        # print(characters)

        for i, (label, image_paths) in enumerate(characters):
            if (i+1) % K == 0 and i!=0: #i+1 % k to make sure that the last K is inside the query set
                # print(i, " here")
                query_images.append(self.image_file_to_array(image_paths, self.dim_input))
                query_labels.append(label)
            else:
                support_images.append(self.image_file_to_array(image_paths, self.dim_input))
                support_labels.append(label)

        # this following 2 lines... I cant be bothered
        support_images, support_labels = np.array(support_images), np.array(support_labels)
        query_images, query_labels = np.array(query_images), np.array(query_labels)

        # print("support_images:", support_images.shape) #(4, 784)=(N*K, image_size)
        # print("support_labels:", support_labels.shape) #(4, 2)=(N*K, N)
        # print("query_images:", query_images.shape) #(2, 784)=(K, image_size)
        # print("query_labels:", query_labels.shape) #(2, 2)=(K, N)

        # Shuffle the query set along the class dimension (dim 1)

        query_indices = np.arange(len(query_images))
        shuffle_fn(query_indices)
        query_images = query_images[query_indices]
        query_labels = query_labels[query_indices]

        images = np.concatenate((support_images, query_images), axis=0) #(K+N, image_size)
        labels = np.concatenate((support_labels, query_labels), axis=0) #(K+N, N)

        images, labels = images.reshape((K, N, -1)), labels.reshape((K, N, N))

        # print("images.shape: ", images.shape)
        # print("labels.shape: ", labels.shape)

        return images, labels
    
    def __iter__(self):
        while True:
            yield self._sample()
