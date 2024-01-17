import click
import logging
from pathlib import Path
from dotenv import find_dotenv, load_dotenv

import os
import random
import shutil


def make_dataset(data_dir, train_dir, val_dir, test_dir, train_percent=0.8, val_percent=0.1):
    """
    Splits a dataset of images into training, validation, and test sets, balancing the classes between them.
    Args:
        data_dir: str, path to the directory containing the original dataset
        train_dir: str, path to the directory where the training set will be saved
        val_dir: str, path to the directory where the validation set will be saved
        test_dir: str, path to the directory where the test set will be saved
        train_percent: float, percentage of images to be used for training (default is 0.8)
        val_percent: float, percentage of images to be used for validation (default is 0.1)
    """
    # Create the train, validation, and test directories if they don't exist
    if not os.path.exists(train_dir):
        os.makedirs(train_dir)
    if not os.path.exists(val_dir):
        os.makedirs(val_dir)
    if not os.path.exists(test_dir):
        os.makedirs(test_dir)

    # Get a list of all the image files in the data directory
    image_files = [f for f in os.listdir(data_dir) if f.endswith(".jpg")]

    # Shuffle the image files randomly
    random.shuffle(image_files)

    # Count the number of images in each class
    class_counts = {}
    for image_file in image_files:
        class_name = image_file.split("-")[0]
        if class_name not in class_counts:
            class_counts[class_name] = 0
        class_counts[class_name] += 1

    # Calculate the number of images to use for training, validation, and test for each class
    train_counts = {}
    val_counts = {}
    test_counts = {}
    for class_name, count in class_counts.items():
        train_counts[class_name] = int(count * train_percent)
        val_counts[class_name] = int(count * val_percent)
        test_counts[class_name] = count - train_counts[class_name] - val_counts[class_name]

    # Copy the images to the train, validation, and test directories while balancing the classes
    for class_name, count in class_counts.items():
        train_count = train_counts[class_name]
        val_count = val_counts[class_name]
        test_count = test_counts[class_name]

        # Get a list of image files for this class
        class_files = [f for f in image_files if f.startswith(class_name)]

        # Split the image files into training, validation, and test sets
        train_files = class_files[:train_count]
        val_files = class_files[train_count:train_count+val_count]
        test_files = class_files[train_count+val_count:]

        # Copy the training files to the train directory
        for train_file in train_files:
            src_path = os.path.join(data_dir, train_file)
            dst_path = os.path.join(train_dir, train_file)
            shutil.copy(src_path, dst_path)

        # Copy the validation files to the validation directory
        for val_file in val_files:
            src_path = os.path.join(data_dir, val_file)
            dst_path = os.path.join(val_dir, val_file)
            shutil.copy(src_path, dst_path)

        # Copy the test files to the test directory
        for test_file in test_files:
            src_path = os.path.join(data_dir, test_file)
            dst_path = os.path.join(test_dir, test_file)
            shutil.copy(src_path, dst_path)

    print("Training set:", train_counts)
    print("Validation set:", val_counts)
    print("Test set:", test_counts)

    return 0

# Example usage
#make_dataset("/content/Meat-Quality-10/train", "/content/Phteven/data/raw/train_split/train",
#             "/content/Phteven/data/raw/train_split/val", "/content/Phteven/data/raw/test")
