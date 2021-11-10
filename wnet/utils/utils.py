# -*- coding: utf-8 -*-

import os
import argparse
import random
import re

import matplotlib.pyplot as plt
import numpy as np
import torch


def list_files_path(path):
    """
    List files from a path.

    :param path: Folder path
    :type path: str
    :return: A list containing all files in the folder
    :rtype: List
    """
    return sorted_alphanumeric(
        [path + f for f in os.listdir(path) if os.path.isfile(os.path.join(path, f))]
    )


def shuffle_lists(lista, listb, seed=42):
    """
    Shuffle two list with the same seed.

    :param lista: List of elements
    :type lista: List
    :param listb: List of elements
    :type listb: List
    :param seed: Seed number
    :type seed: int
    :return: lista and listb shuffled
    :rtype: (List, List)
    """
    random.seed(seed)
    random.shuffle(lista)
    random.seed(seed)
    random.shuffle(listb)
    return lista, listb


def shuffle_list(lista, seed=42):
    """
    Shuffle two list with the same seed.

    :param lista: List of elements
    :type lista: List
    :param listb: List of elements
    :type listb: List
    :param seed: Seed number
    :type seed: int
    :return: lista and listb shuffled
    :rtype: (List, List)
    """
    random.seed(seed)
    random.shuffle(lista)
    return lista


def print_red(skk):
    """
    Print in red.

    :param skk: Str to print
    :type skk: str
    """
    print("\033[91m{}\033[00m".format(skk))


def print_gre(skk):
    """
    Print in green.

    :param skk: Str to print
    :type skk: str
    """
    print("\033[92m{}\033[00m".format(skk))


def sorted_alphanumeric(data):
    """
    Sort function.

    :param data: str list
    :type data: List
    :return: Sorted list
    :rtype: List
    """
    convert = lambda text: int(text) if text.isdigit() else text.lower()  # noqa
    alphanum_key = lambda key: [convert(c) for c in re.split("([0-9]+)", key)]  # noqa
    return sorted(data, key=alphanum_key)


def learning_curves(train_enc, train_recons, val_enc, val_recons):
    fig = plt.figure(figsize=(15, 10))
    ax = []
    ax.append(fig.add_subplot(1, 2, 1))
    ax.append(fig.add_subplot(1, 2, 2))
    fig.suptitle("Training Curves")
    ax[0].plot(train_enc, label="Train Enc")
    ax[0].plot(val_enc, label="Validation Enc")
    ax[1].plot(train_recons, label="Train Recons")
    ax[1].plot(val_recons, label="Validation Recons")
    ax[0].set_ylabel("Loss value", fontsize=14)
    ax[0].set_xlabel("Epoch", fontsize=14)
    ax[1].set_ylabel("Loss value", fontsize=14)
    ax[1].set_xlabel("Epoch", fontsize=14)
    ax[0].legend(loc="upper right")
    ax[1].legend(loc="upper right")
    fig.savefig("data/plot.png")
    plt.close(fig)


def get_args():
    """
    Argument parser.

    :return: Object containing all the parameters needed to train a model
    :rtype: Dict
    """
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--epochs", "-e", type=int, default=100, help="number of epochs of training"
    )
    parser.add_argument(
        "--batch_size", "-bs", type=int, default=1, help="size of the batches"
    )
    parser.add_argument("--lr", type=float, default=0.001, help="adam: learning rate")
    parser.add_argument(
        "--size", type=int, default=512, help="Size of the image, one number"
    )
    parser.add_argument("--drop_r", "-d", type=float, default=0.2, help="Dropout rate")
    parser.add_argument(
        "--filters", "-f",
        type=int,
        default=8,
        help="Number of filters in first conv block",
    )
    args = parser.parse_args()
    print_red(args)
    return args


def visualize(net, image, k, opt):
    if k % 2 == 0 or k == 1:
        mask, att = net.forward_enc(image)
        output = net.forward(image)
        image = (image.cpu().numpy() * 255).astype(np.uint8).reshape(-1, opt.size, opt.size)
        argmax = torch.argmax(mask, 1)
        pred, output = (
            (argmax.detach().cpu() * 255).numpy().astype(np.uint8),
            (output.detach().cpu() * 255).numpy().astype(np.uint8).reshape(-1, opt.size, opt.size),
        )
        plot_images(image, pred, att.detach().cpu(), output, k, opt.size)


def plot_images(imgs, pred, att, output, k, size):
    fig = plt.figure(figsize=(15, 10))
    columns = 4
    rows = 5  # nb images
    ax = []  # loop around here to plot more images
    i = 0
    for j, img in enumerate(imgs):
        ax.append(fig.add_subplot(rows, columns, i + 1))
        ax[-1].set_title("Input")
        plt.imshow(img, cmap="gray")

        ax.append(fig.add_subplot(rows, columns, i + 2))
        ax[-1].set_title("Mask")
        plt.imshow(pred[j].reshape((size, size)), cmap="gray")

        ax.append(fig.add_subplot(rows, columns, i + 3))
        ax[-1].set_title("Attention Map")
        plt.imshow(att[j].reshape((size, size)))
        plt.colorbar()

        ax.append(fig.add_subplot(rows, columns, i + 4))
        ax[-1].set_title("Output")
        plt.imshow(output[j].reshape((size, size)), cmap="gray")

        i += 4
        if i >= 15:
            break
    plt.savefig("data/results/epoch_" + str(k) + ".png")
    plt.close()
