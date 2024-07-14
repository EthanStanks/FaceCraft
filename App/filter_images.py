import cv2
from PIL import Image
import numpy as np

def cartoonify(img):
    numpy_img = np.array(img)
    grayScaleImage = cv2.cvtColor(numpy_img, cv2.COLOR_RGB2GRAY)
    smoothGrayScale = cv2.medianBlur(grayScaleImage, 5)
    getEdge = cv2.adaptiveThreshold(smoothGrayScale, 255, 
                                    cv2.ADAPTIVE_THRESH_MEAN_C, 
                                    cv2.THRESH_BINARY, 9, 9)
    colorImage = cv2.bilateralFilter(numpy_img, 9, 300, 300)
    cartoonImage = cv2.bitwise_and(colorImage, colorImage, mask=getEdge)
    cartoonImage = Image.fromarray(cartoonImage)
    return cartoonImage

def pencil(img):
    numpy_img = np.array(img)
    gray_img = cv2.cvtColor(numpy_img, cv2.COLOR_BGR2GRAY)
    blurred_img = cv2.GaussianBlur(gray_img, (21, 21), sigmaX=0, sigmaY=0)
    penicl = cv2.divide(gray_img, blurred_img, scale=256)
    penicl = Image.fromarray(penicl)
    return penicl

def oil_painting(img):
    numpy_img = np.array(img)
    oil_painting = cv2.xphoto.oilPainting(numpy_img, 7, 1)
    oil_painting = Image.fromarray(oil_painting)
    return oil_painting

def watercolor(img):
    numpy_img = np.array(img)
    median = cv2.medianBlur(numpy_img, 15)
    watercolor = cv2.bilateralFilter(median, d=75, sigmaColor=75, sigmaSpace=75)
    watercolor = Image.fromarray(watercolor)
    return watercolor

def black_and_white(img):
    numpy_img = np.array(img)
    gray_img = cv2.cvtColor(numpy_img, cv2.COLOR_BGR2GRAY)
    gray_img = Image.fromarray(gray_img)
    return gray_img

def sepia(img):
    numpy_img = np.array(img)
    sepia_filter = np.array([[0.272, 0.534, 0.131],
                             [0.349, 0.686, 0.168],
                             [0.393, 0.769, 0.189]])
    sepia_img = cv2.transform(numpy_img, sepia_filter)
    sepia_img = np.clip(sepia_img, 0, 255).astype(np.uint8)
    sepia_img = Image.fromarray(sepia_img)
    return sepia_img

def blue_tone(img):
    numpy_img = np.array(img)
    blue_tone_img = numpy_img.copy()
    blue_tone_img[:, :, 0] = cv2.add(blue_tone_img[:, :, 0], 50)
    blue_tone_img = np.clip(blue_tone_img, 0, 255)
    blue_tone_img = Image.fromarray(blue_tone_img)
    return blue_tone_img

def xray_effect(img):
    numpy_img = np.array(img)
    gray_img = cv2.cvtColor(numpy_img, cv2.COLOR_BGR2GRAY)
    xray_img = 255 - gray_img
    xray_img = Image.fromarray(xray_img)
    return xray_img