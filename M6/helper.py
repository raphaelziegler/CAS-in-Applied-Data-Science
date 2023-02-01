#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# File name: helper.py

# imports
import os
from PIL import Image
import numpy as np


def get_image_path():

    # create empty variables to store file paths
    rhino_path = []
    buffalo_path = []
    elephant_path = []
    zebra_path = []
    # preface = "img/"
    preface = "./"

    # read folders in the img folder
    img_cat = os.listdir(preface)  # list with filenames

    # loop through specific folders
    for cat in img_cat:
        # print(cat)
        if cat == "rhino":
            # add jpg files to variable
            rhino_path = [f"{preface}rhino/{item}" for item in os.listdir(f"{preface}{cat}") if
                          item[-3:] == "jpg" or item[-3:] == "JPG"]
        elif cat == "buffalo":
            buffalo_path = [f"{preface}buffalo/{item}" for item in os.listdir(f"{preface}{cat}") if
                            item[-3:] == "jpg" or item[-3:] == "JPG"]
        elif cat == "elephant":
            elephant_path = [f"{preface}elephant/{item}" for item in os.listdir(f"{preface}{cat}") if
                             item[-3:] == "jpg" or item[-3:] == "JPG"]
        elif cat == "zebra":
            zebra_path = [f"{preface}zebra/{item}" for item in os.listdir(f"{preface}{cat}") if
                          item[-3:] == "jpg" or item[-3:] == "JPG"]

    return rhino_path, buffalo_path, elephant_path, zebra_path


def image_stuff(path, width=200, height=200, grey=False):

    # read image and convert to greyscale if wanted
    if grey is True:
        img = Image.open(path).convert("L")
    else:
        img = Image.open(path)

    # resize image
    img_resized = img.resize((width, height))

    return np.array(img_resized, dtype=np.float_)


if __name__ == '__main__':

    rp, bp, ep, zp = get_image_path()

    # print("rhino: ", len(rp))
    # print("buffalo: ", len(bp))
    # print("elephant: ", len(ep))
    # print("zebra: ", len(zp))

    print(rp, bp, ep, zp)

    print(rp[0])
    i = image_stuff(rp[0], grey=False)
    print(i)