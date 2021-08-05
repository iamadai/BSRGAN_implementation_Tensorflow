import os
import tensorflow as tf
import random

from unprocess import unprocess, random_noise_levels, add_noise
from process import process

import tensorflow_addons as tfa
from tensorflow.python.data.experimental import AUTOTUNE

def check_image_file(filename: str):
    r"""Filter non image files in directory.

    Args:
        filename (str): File name under path.

    Returns:
        Return True if bool(x) is True for any x in the iterable.
    """
    return any(filename.endswith(extension) for extension in [".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".JPG", ".JPEG", ".PNG", ".BMP"])

class Data:
    def __init__(self,
                 scale=2,
                 subset='train',
                 images_dir='videos/images',
                 caches_dir='videos/caches'):

        self.scale = scale
        self.subset = subset
        self.hr_crop_size = 160

        self.HR_fold = f'{self.subset}'

        self.images_dir = images_dir
        self.caches_dir = caches_dir

        os.makedirs(images_dir, exist_ok=True)
        os.makedirs(caches_dir, exist_ok=True)

        hr_root = self._hr_images_dir()
        self.HR_filenames = [os.path.join(hr_root, x) for x in os.listdir(hr_root) if check_image_file(x)]

    def __len__(self):
        return len(self.HR_filenames)

    def dataset(self, batch_size=16, repeat_count=None):
        ds = tf.data.Dataset.zip(self.hr_dataset())
        ds = ds.map(lambda hr: random_crop(hr, hr_crop_size=self.hr_crop_size), num_parallel_calls=AUTOTUNE)
        ds = ds.map(lambda hr: generate_pair(hr, hr_crop_size=self.hr_crop_size, scale=self.scale), num_parallel_calls=AUTOTUNE)
        ds = ds.batch(batch_size)
        ds = ds.repeat(repeat_count)
        ds = ds.prefetch(buffer_size=AUTOTUNE)
        return ds

    def hr_dataset(self):
        ds = self._images_dataset(self.HR_filenames).cache(self._hr_cache_file())
        if not os.path.exists(self._hr_cache_index()):
            self._populate_cache(ds, self._hr_cache_file())

        return ds

    def _hr_cache_file(self):
        return os.path.join(self.caches_dir, f'{self.subset}_HR.cache')

    def _hr_cache_index(self):
        return f'{self._hr_cache_file()}.index'

    def _hr_images_dir(self):
        return os.path.join(self.images_dir, self.HR_fold)

    @staticmethod
    def _images_dataset(image_files):
        ds = tf.data.Dataset.from_tensor_slices(image_files)
        ds = ds.map(tf.io.read_file)
        ds = ds.map(lambda x: tf.image.decode_png(x, channels=3), num_parallel_calls=AUTOTUNE)
        return ds

    @staticmethod
    def _populate_cache(ds, cache_file):
        print(f'Caching decoded images in {cache_file} ...')
        for _ in ds: pass
        print(f'Cached decoded images in {cache_file}.')

def random_crop(hr_img, hr_crop_size=160):
    hr_img_cropped = tf.image.random_crop(value=hr_img, size=(hr_crop_size, hr_crop_size, 3))
    return hr_img_cropped

def generate_pair(hr_img, hr_crop_size=160, scale=2):
    lr_img = hr_img
    degrade_seq = ["down", "blur_iso", "blur_aniso", "noise", "camera"]
    if random.randint(1, 4) <= 3:
        degrade_seq.append("jpeg")
    random.shuffle(degrade_seq)
    degrade_seq.append("jpeg")

    lr_w = hr_crop_size
    lr_h = hr_crop_size
    for mode in degrade_seq:
        if mode == "down":
            lr_img = down_image(lr_img, hr_crop_size=hr_crop_size, scale=scale)
            lr_w = hr_crop_size // scale
            lr_h = hr_crop_size // scale
        elif mode == "camera":
            lr_img = camera_effect(lr_img)
        elif mode == "blur_iso":
            lr_img = blur_effect(lr_img, is_aniso=False)
        elif mode == "blur_aniso":
            lr_img = blur_effect(lr_img, is_aniso=True)
        elif mode == "noise":
            lr_img = noise_effect(lr_img)
        elif mode == "jpeg":
            lr_img.set_shape([lr_h, lr_w, 3])
            lr_img = tf.image.random_jpeg_quality(lr_img, min_jpeg_quality=65, max_jpeg_quality=95)

    return lr_img, hr_img

def blur_effect(img, is_aniso=False):
    filter_shape = random.choice([7, 9, 11, 13, 15, 17, 19, 21])
    if is_aniso:
        sigma = random.uniform(0.1, 2.8)
    else:
        sigma = [random.uniform(0.5, 8), random.uniform(0.5, 8)]

    processed = tfa.image.gaussian_filter2d(image=img, filter_shape=[filter_shape, filter_shape], sigma=sigma)
    return processed

def down_image(hr_img, hr_crop_size=160, scale=2):
    img_shape = tf.shape(hr_img)[:2]
    lr_w = img_shape[1] // scale
    lr_h = img_shape[0] // scale
    mode = random.randint(1, 8)
    method = tf.image.ResizeMethod.NEAREST_NEIGHBOR
    if mode == 1:
        method = tf.image.ResizeMethod.NEAREST_NEIGHBOR
    elif mode == 2:
        method = tf.image.ResizeMethod.BILINEAR
    elif mode == 3:
        method = tf.image.ResizeMethod.BICUBIC
    elif mode == 4:
        method = tf.image.ResizeMethod.GAUSSIAN
    elif mode == 5:
        method = tf.image.ResizeMethod.AREA
    elif mode == 6:
        method = tf.image.ResizeMethod.LANCZOS3
    elif mode == 7:
        method = tf.image.ResizeMethod.LANCZOS5
    elif mode == 8:
        method = tf.image.ResizeMethod.MITCHELLCUBIC

    hr_img.set_shape([hr_crop_size, hr_crop_size, 3])
    lr_img = tf.image.resize(hr_img, size=[lr_h, lr_w], method=method, antialias=True)
    return lr_img

def noise_effect(image):
    noise_level = random.randint(1, 25)
    img_shape = tf.shape(image)
    noise = tf.random.normal(img_shape, mean=0, stddev=noise_level)
    noise_img = tf.cast(image, tf.float32)
    noise_img = tf.math.add(noise_img, noise)
    noise_img = tf.experimental.numpy.clip(noise_img, 0., 255.)
    noise_img = tf.cast(noise_img, tf.uint8)
    return noise_img

def camera_effect(img):
    img = tf.cast(img, tf.float32) / 255.
    deg_img, features = unprocess(img)

    shot_noise, read_noise = random_noise_levels()
    deg_img = add_noise(deg_img, shot_noise, read_noise)

    deg_img = tf.expand_dims(deg_img, 0)
    features['red_gain'] = tf.expand_dims(features['red_gain'], axis=0)
    features['blue_gain'] = tf.expand_dims(features['blue_gain'], axis=0)
    features['cam2rgb'] = tf.expand_dims(features['cam2rgb'], axis=0)
    deg_img = process(deg_img, features['red_gain'], features['blue_gain'], features['cam2rgb'])
    deg_img = tf.squeeze(deg_img)
    deg_img = tf.saturate_cast(deg_img * 255 + 0.5, tf.uint8)
    return deg_img
