#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# File name: image_manipulation.py

# imports
import os
from PIL import Image
import numpy as np
from matplotlib import pyplot as plt
import tensorflow as tf


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

def read_metadata_file(img_path):

    # create path to text file
    metadata_path = img_path[:-3] + "txt"

    with open(metadata_path, "r") as file:
        objects = []
        for line in file.readlines():
            objects.append(line.strip())
    return objects

def image_stuff(path, objects=[], w=200, h=200):
    img_out = []
    width = 400
    height = 400

    # crop and resize image
    # center is the object in the object file, file gets padding if necessary

    # do we have object data
    if len(objects) > 0:
        for object in objects:
            # print("cropping")
            # read image
            img = Image.open(path)
            # get dimensions
            W, H = img.size

            # read metadata
            classification, cw, ch, dw, dh = object.split(" ")
            # print(classification, cw, ch, dw, dh)

            # check if padding is needed
            cw = float(cw) * W
            ch = float(ch) * H
            dw = float(dw) * W
            dh = float(dh) * H


            if dw > w:
                width = dw
            if dh > h:
                height = dh
            width = max(width, height)
            height = width
            # print(width, height)

            # print(cw, ch)


            pl, pr, pt, pb = 0, 0, 0, 0
            if width / 2 - cw > 0:
                pl = width / 2 - cw
            if cw + width / 2 - W > 0:
                pr = cw + width / 2 - W
            if ch + height / 2 - H > 0:
                pt = ch + height / 2 - H
            if height / 2 - ch > 0:
                pb = height / 2 - ch

            new_W = int(W + pr + pl)
            new_H = int(H + pt + pb)

            # print(new_W, new_H)

            """
            color = tuple(np.random.choice(range(256), size=3))
            p_img = Image.new(img.mode, (new_W, new_H), color)
            p_img.paste(img, (int(pl), int(pt)))
            """

            # crop
            left = int(cw - width / 2)
            right = left + width
            top = int(ch - height / 2)
            bottom = top + height

            # print(left, top, right, bottom)

            c_img = img.crop((left, top, right, bottom))  # p_img


            # resize
            r_img = c_img.resize((w, h))  # c_img
            img_out.append(r_img)

            # img.show()
            # p_img.show()
            # c_img.show()

    else:
        print(f"doing nothing: {path}")

    # img_out = np.array(img_out)
    # img_out = img_out.astype(np.float32)
    # img_preproc = tf.keras.applications.nasnet.preprocess_input(img_out)
    
    return img_out

def arr_to_img(arr):
    plt.imshow(arr, interpolation='nearest')
    plt.show()


if __name__ == '__main__':

    rp, bp, ep, zp = get_image_path()

    # print("rhino: ", len(rp))
    # print("buffalo: ", len(bp))
    # print("elephant: ", len(ep))
    # print("zebra: ", len(zp))

    # print(rp, bp, ep, zp)

    # print(rp[0])
    # i = image_stuff(rp[0], grey=False)
    # print(i)

    for path in rp:
        print(path)
        metadata = read_metadata_file(path)
        print(metadata)
        p_i = image_stuff(path, metadata, 300, 300)
        pic_arrs = []
        for i in p_i:
            # i.show()
            # print(np.array(i))
            pic_arrs.append(np.array(i))
        # print(pic_arrs)
        for pic in pic_arrs:
            print(len(pic.shape))
            arr_to_img(pic)
