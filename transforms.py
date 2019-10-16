from .imports import *

def random_pad_crop(x: tf.Tensor, pad_size: int) -> tf.Tensor:
  """
    Randomly pad the image by `pad_size` at each border (top, bottom, left, right). Then, crop the padded image to its
    original size.

    :param x: Input image.
    :param pad_size: Number of pixels to pad at each border. For example, a 32x32 image padded with 4 pixels becomes a
                     40x40 image. Then, the subsequent cropping step crops the image back to 32x32. Padding is done in
                     `reflect` mode.
    :return: Transformed image.
  """
  shape = tf.shape(x)
  x = tf.pad(x, [[pad_size, pad_size], [pad_size, pad_size], [0, 0]], mode='reflect')
  x = tf.random_crop(x, [shape[0], shape[1], 3])
  return x

def random_flip(x: tf.Tensor, flip_vert: bool = False) -> tf.Tensor:
  """
    Randomly flip the input image horizontally, and optionally also vertically, which is implemented as 90-degree
    rotations.
    :param x: Input image.
    :param flip_vert: Whether to perform vertical flipping. Default: False.
    :return: Transformed image.
  """
  x = tf.image.random_flip_left_right(x)
  if flip_vert:
    x = random_rotate_90(x)
  return x

def replace_slice(input_: tf.Tensor, replacement, begin) -> tf.Tensor:
    inp_shape = tf.shape(input_)
    size = tf.shape(replacement)
    padding = tf.stack([begin, inp_shape - (begin + size)], axis=1)
    replacement_pad = tf.pad(replacement, padding)
    mask = tf.pad(tf.ones_like(replacement, dtype=tf.bool), padding)
    return tf.where(mask, replacement_pad, input_)
  
def cutout(x: tf.Tensor, h: int, w: int, c: int = 3) -> tf.Tensor:
  """
    Cutout data augmentation. Randomly cuts a h by w whole in the image, and fill the whole with zeros.
    :param x: Input image.
    :param h: Height of the hole.
    :param w: Width of the hole
    :param c: Number of color channels in the image. Default: 3 (RGB).
    :return: Transformed image.
  """
  shape = tf.shape(x)
  x0 = tf.random.uniform([], 0, shape[0] + 1 - h, dtype=tf.int32)
  y0 = tf.random.uniform([], 0, shape[1] + 1 - w, dtype=tf.int32)
  x = replace_slice(x, tf.zeros([h, w, c]), [x0, y0, 0])
  return x