import cv2
import numpy as np
import glob

image_address = glob.glob('data/p10/day01/*.jpg')

from PIL import Image, ImageChops

im = Image.open(image_address[0])


def trim(im):
    bg = Image.new(im.mode, im.size, im.getpixel((0, 0)))
    diff = ImageChops.difference(im, bg)
    diff = ImageChops.add(diff, diff, 2.0, -100)
    bbox = diff.getbbox()
    if bbox:
        return im.crop(bbox)


print(np.array(trim(im)).shape)
