#!python
# -*- coding: utf-8 -*-

__author__ = "Tomas Zitka"
__email__ = "zitkat@kky.zcu.cz"


from PIL import Image, ImageDraw, ImageFont
import cv2
import numpy as np
import matplotlib.pyplot as plt

from functools import wraps


def returns_pil(f):
    @wraps(f)
    def returnpil(*args, nopil=False, **kwargs):
        im = f(*args, **kwargs)
        if nopil:
            return im
        return Image.fromarray(np.uint8(255 * im))

    return returnpil


def add_cam_overlay(img: np.ndarray,
                    mask: np.ndarray,
                    use_rgb: bool = True,
                    colormap: int = cv2.COLORMAP_INFERNO) -> np.ndarray:
    """ This function  overlays the cam mask on the image as a heatmap.
    By default the heatmap is in BGR format.

    Accept arguments in range [0, 1].

    :param img: The base image in RGB or BGR format.
    :param mask: The cam mask.
    :param use_rgb: Whether to use an RGB or BGR heatmap,
        this should be set to True if 'img' is in RGB format.
    :param colormap: The OpenCV colormap to be used.
    :returns: The default image with the cam overlay.
    """
    heatmap = cv2.applyColorMap(np.uint8(255 * mask), colormap)
    if use_rgb:
        heatmap = cv2.cvtColor(heatmap, cv2.COLOR_BGR2RGB)
    heatmap = np.float32(heatmap) / 255

    if np.max(img) > 1:
        raise Exception(
                "The input image should np.float32 in the range [0, 1]")

    cam = heatmap + img
    cam = cam / np.max(cam)
    return cam


def tile_image_with_preds(img, predictions):
    img_tiled = []
    for pred in predictions:
        img_tiled.append(prepend_char_to_img(pred, img))
    return np.concatenate(img_tiled)


def prepend_char_to_img(pred: str, img: np.ndarray, font_path="Courier_Prime.ttf"):
    """
    Takes RGB image in range [0,1] and one character string and renders character to the left
    of the image
    """
    height = img.shape[0]
    width = img.shape[1]
    new_img = np.ones((height, width + height, 3))
    new_img[:, height:] = img

    pimg = Image.fromarray((255 * new_img).astype(np.uint8))
    dwarf = ImageDraw.Draw(pimg)
    dwarf.text((2, 0), f"{pred.replace(' ', '_')}", fill=(0, 155, 0),
               font=ImageFont.truetype(font_path, size=64))

    return np.array(pimg) / 255.


def prepend_col_string_to_img(string: str, img: np.ndarray, col_width=64, font_path="Courier_Prime.ttf") -> np.ndarray:
    height, width, _ = img.shape
    new_img = np.ones((height, width + col_width, 3), dtype=np.uint8) * 255
    new_img[:, col_width:] = (255 * img).astype(np.uint8)

    pimg = Image.fromarray(new_img)
    dwarf = ImageDraw.Draw(pimg)
    for ii, c in enumerate(string):
        dwarf.text((2, ii * col_width), f"{c.replace(' ', '_')}", fill=(0, 0, 0),
                   font=ImageFont.truetype(font_path, size=65))
    return np.array(pimg) / 255


def draw_spaced_string(prediction, height, font_path="Courier_Prime.ttf"):
    prediction_img = Image.fromarray(np.full((height, len(prediction) * height, 3),
                                             255,
                                     dtype=np.uint8))
    pdraw = ImageDraw.Draw(prediction_img)

    for ii, c in enumerate(prediction):
        pdraw.text((2 + ii*height, 0), c, fill=(0, 0, 0),
                   font=ImageFont.truetype(font_path, size=64))
    return np.array(prediction_img) / 255


@returns_pil
def create_text2image_coattention_heatmap(co_attention: np.ndarray,
                                          prediction: str,
                                          input_img: np.ndarray, data_width: int,
                                          show_text=True) -> np.ndarray:
    height, width, _ = input_img.shape

    attention_stride = int(np.ceil(width / co_attention.shape[-1]))
    co_attention = np.repeat(co_attention, attention_stride, axis=1)
    co_attention_heatmap = np.repeat(co_attention, height, axis=0)[:, :input_img.shape[-2]]

    tiled_img = np.tile(input_img, (len(prediction) + 1, 1, 1))

    cam_image = add_cam_overlay(tiled_img, co_attention_heatmap)
    cam_image = cam_image[:, :data_width]
    if show_text:
        cam_image = prepend_col_string_to_img(prediction + '\n', cam_image)

    return cam_image


@returns_pil
def create_image_selfattention_heatmap(self_attention: np.ndarray,
                                       input_img: np.ndarray, data_width: int) -> np.ndarray:
    height, width = input_img.shape[:2]
    attention_stride = int(np.ceil(width / self_attention.shape[-1]))
    self_attention = np.repeat(np.repeat(self_attention,
                                         attention_stride, axis=0),
                               attention_stride, axis=1)
    self_attention = self_attention[:width, :width]

    cam_image = np.zeros(2 * (height + width,) + (3,))
    cam_image[:height, height:, :] = input_img
    cam_image[height:, :height, :] = input_img.swapaxes(0, 1)
    cam_image[height:, height:, :] = add_cam_overlay(cam_image[height:, height:],
                                                     self_attention)

    cam_image = cam_image[:height + data_width, : height + data_width]

    for px in range(attention_stride + height, data_width + height, attention_stride):
        cv2.line(cam_image, (px, 0), (px, data_width + height), (0, 0, 0), thickness=1)
        cv2.line(cam_image, (0, px), (data_width + height, px), (0, 0, 0), thickness=1)

    return cam_image


@returns_pil
def create_text_selfattention_heatmap(prediction: str, self_attention: np.ndarray, height=64):

    prediction_img = draw_spaced_string(prediction + "\n", height=height)
    prediction_img_tiled = tile_image_with_preds(prediction_img, prediction + "\n")
    attention_stride = height
    self_attention_tiled = np.repeat(np.repeat(self_attention, attention_stride, axis=0),
                                     attention_stride, axis=1)

    cam_image = prediction_img_tiled.copy()
    cam_image[:, height:] = add_cam_overlay(cam_image[:, height:], self_attention_tiled)
    return cam_image
