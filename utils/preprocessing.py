import glob
import os
import re
import numpy as np
import matplotlib.pyplot as plt
import PIL
from PIL import Image


class Preprocessor(object):
    DATASET_DIR = '../input/'

    @staticmethod
    def natural_key(string_):
        """
        Define sort key that is integer-aware
        """
        return [int(s) if s.isdigit() else s for s in re.split(r'(\d+)', string_)]

    @staticmethod
    def norm_image(image):
        """
        Normalize PIL image

        Normalizes luminance to (mean,std)=(0,1), and applies a [1%, 99%] contrast stretch
        """
        img_y, img_b, img_r = image.convert('YCbCr').split()

        img_y_np = np.asarray(img_y).astype(float)

        img_y_np /= 255
        img_y_np -= img_y_np.mean()
        img_y_np /= img_y_np.std()
        scale = np.max([np.abs(np.percentile(img_y_np, 1)), np.abs(np.percentile(img_y_np, 99))])

        img_y_np /= scale
        img_y_np = np.clip(img_y_np, -1.0, 1.0)
        img_y_np = (img_y_np + 1.0) / 2.0

        img_y_np = (img_y_np * 255 + 0.5).astype(np.uint8)
        img_y = Image.fromarray(img_y_np)
        img_ybr = Image.merge('YCbCr', (img_y, img_b, img_r))

        img_nrm = img_ybr.convert('RGB')
        return img_nrm

    @staticmethod
    def resize_image(image, size):
        """
        Resize PIL image

        Resizes image to be square with sidelength size. Pads with black if needed.
        """
        # Resize
        n_x, n_y = image.size
        if n_y > n_x:
            n_y_new = size
            n_x_new = int(size * n_x / n_y + 0.5)
        else:
            n_x_new = size
            n_y_new = int(size * n_y / n_x + 0.5)

        img_res = image.resize((n_x_new, n_y_new), resample=PIL.Image.BICUBIC)

        # Pad the borders to create a square image
        img_pad = Image.new('RGB', (size, size), (128, 128, 128))
        ulc = ((size - n_x_new) // 2, (size - n_y_new) // 2)
        img_pad.paste(img_res, ulc)

        return img_pad

    @staticmethod
    def get_dataset_paths(dataset_dir=DATASET_DIR, data_regex='*.JPEG', key=natural_key.__func__):
        dataset = sorted(glob.glob(os.path.join(dataset_dir, data_regex)), key=key)

        return dataset

    @staticmethod
    def get_dataset_paths_from_wnid(wnids):
        dataset = []
        for wnid in wnids:
            dataset_dir = Preprocessor.DATASET_DIR + wnid + '/'
            image_paths = Preprocessor.get_dataset_paths(dataset_dir)
            dataset.extend(image_paths)

        return dataset

    @staticmethod
    def get_image(image_path):
        return Image.open(image_path)

    @staticmethod
    def get_processed_image(image, size=224):
        image_normalized = Preprocessor.norm_image(image)
        image_resized = Preprocessor.resize_image(image_normalized, size)
        return image_resized

    @staticmethod
    def get_wnid_table(table_directory='../input/wnid_to_labels.txt'):
        wnid_table = dict()
        with open(table_directory, 'r') as wnid_table_file:
            for line in wnid_table_file:
                wnid_id, lable = line.split('\t')
                wnid_table[wnid_id] = lable.strip()

        return wnid_table

    @staticmethod
    def get_dataset_wnids(dataset_dir=DATASET_DIR):
        wnids_paths = glob.glob(os.path.join(dataset_dir + '*'))
        wnids_regex = lambda wnid_path: re.search(r"/(n.*)", wnid_path)

        wnids = [wnids_regex(wnids_path).group(1) for wnids_path in wnids_paths if wnids_regex(wnids_path)]
        return wnids

    @staticmethod
    def extract_wnid(wnid_path):
        wnids_regex = lambda wnid_path: re.search(r"/(n.*)/", wnid_path)

        return wnids_regex(wnid_path).group(1) if wnids_regex(wnid_path) else None

    @staticmethod
    def create_wnid_to_one_hot(wnids):
        class_num = len(wnids)
        one_hot_matrix = np.eye(class_num)

        wnid_to_one_hot = dict()
        for i, wnid in enumerate(wnids):
            wnid_to_one_hot[wnid] = one_hot_matrix[i]

        return wnid_to_one_hot

    @staticmethod
    def get_processed_image_from_path(path, image_size=64):
        image = Preprocessor.get_image(path)
        processed_image = Preprocessor.get_processed_image(image, size=image_size)
        return np.array(processed_image)


class Analytics(object):

    @staticmethod
    def extract_class_distribution(dataset):
        class_counter = dict()
        for image_path in dataset:
            class_wnid = Preprocessor.extract_wnid(image_path)
            class_counter[class_wnid] = class_counter.get(class_wnid, 0) + 1

        return class_counter

    @staticmethod
    def set_image_size(width=18.5, height=10.5, dpi=160):
        fig = plt.gcf()
        fig.set_size_inches(width, height)
        fig.set_dpi(dpi)
