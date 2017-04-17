import glob
import shutil
import os
import itertools
import re
import numpy as np
import random
import matplotlib.pyplot as plt
import PIL
from PIL import Image

from sklearn.metrics import confusion_matrix


class ImageProcessor(object):

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
    def get_image(image_path):
        return Image.open(image_path)

    @staticmethod
    def get_processed_image(image, size=224):
        image_normalized = ImageProcessor.norm_image(image)
        image_resized = ImageProcessor.resize_image(image_normalized, size)
        return image_resized

    @staticmethod
    def set_image_size(width=18.5, height=10.5, dpi=160):
        fig = plt.gcf()
        fig.set_size_inches(width, height)
        fig.set_dpi(dpi)

    @staticmethod
    def get_processed_image_from_path(path, image_size=64):
        image = ImageProcessor.get_image(path)
        processed_image = ImageProcessor.get_processed_image(image, size=image_size)
        return np.array(processed_image)


class DatasetProcessor(object):
    DATASET_DIR = '../input'
    TRAIN_DIR = DATASET_DIR + '/train'
    TEST_DIR = DATASET_DIR + '/validation'

    @staticmethod
    def __get_percent_of_files_within_directory(filename, percent=0.1):
        filenames = os.listdir(os.path.join(DatasetProcessor.TRAIN_DIR, filename))
        num_filenames = float(len(filenames))

        print(num_filenames)

        new_files = []
        while ((len(new_files) + 1) / num_filenames) < percent:
            random_filename = random.choice(filenames)

            if random_filename not in new_files:
                new_files.append(random_filename)

        return new_files

    @staticmethod
    def __move_files(filenames, src=TRAIN_DIR, des=TEST_DIR):
        for filename in filenames:
            shutil.move(os.path.join(src, filename), os.path.join(des, filename))

    @staticmethod
    def file_train_test_split(train_dir=TRAIN_DIR, test_dir=TEST_DIR, test_size=0.1):
        print('Moved from dirs:')
        for filename in os.listdir(train_dir):
            dir = os.path.join(DatasetProcessor.TEST_DIR, filename)
            os.makedirs(dir)
            print('  %s' % filename)

            if os.path.isdir(os.path.join(DatasetProcessor.TRAIN_DIR, filename)):
                test_files = DatasetProcessor.__get_percent_of_files_within_directory(filename, test_size)

                print('     %d' % len(test_files))
                src = os.path.join(train_dir, filename)
                des = os.path.join(test_dir, filename)
                DatasetProcessor.__move_files(test_files, src, des)


    @staticmethod
    def natural_key(string_):
        """
        Define sort key that is integer-aware
        """
        return [int(s) if s.isdigit() else s for s in re.split(r'(\d+)', string_)]

    @staticmethod
    def get_dataset_paths(dataset_dir=DATASET_DIR, data_regex='*.JPEG', key=natural_key.__func__):
        dataset = sorted(glob.glob(os.path.join(dataset_dir, data_regex)), key=key)

        return dataset

    @staticmethod
    def get_dataset_paths_from_wnid(wnids, dataset_path=DATASET_DIR):
        dataset = []
        for wnid in wnids:
            dataset_dir = dataset_path + wnid + '/'
            image_paths = DatasetProcessor.get_dataset_paths(dataset_dir)
            dataset.extend(image_paths)

        return dataset

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
    def wnid_to_one_hot(wnids):
        class_num = len(wnids)
        one_hot_matrix = np.eye(class_num)

        wnid_to_one_hot = dict()
        for i, wnid in enumerate(wnids):
            wnid_to_one_hot[wnid] = one_hot_matrix[i]

        return wnid_to_one_hot

    @staticmethod
    def paths_to_one_hot(paths, wnid_one_hot):
        paths = sorted(paths)

        one_hot_mapper = lambda path: wnid_one_hot[DatasetProcessor.extract_wnid(path)]
        return np.array(list(map(one_hot_mapper, paths)))

    @staticmethod
    def softmax_predictions_to_one_hot(predicions):
        class_num = predicions.shape[1]
        one_hot_predictions = []

        for prediction in predicions:
            one_hot_matrix = np.zeros(class_num)
            class_index = np.argmax(prediction)
            one_hot_matrix[class_index] = 1
            one_hot_predictions.append(one_hot_matrix)

        return np.array(one_hot_predictions)

    @staticmethod
    def from_one_hot_to_categorical(one_hot_dataset):
        return [np.argmax(row) for row in one_hot_dataset]


class Analytics(object):

    @staticmethod
    def extract_class_distribution(dataset):
        class_counter = dict()
        for image_path in dataset:
            class_wnid = DatasetProcessor.extract_wnid(image_path)
            class_counter[class_wnid] = class_counter.get(class_wnid, 0) + 1

        return class_counter

    @staticmethod
    def plot_confusion_matrix(cm, classes,
                              normalize=False,
                              title='Confusion matrix',
                              cmap=plt.cm.Blues):
        """
        This function prints and plots the confusion matrix.
        Normalization can be applied by setting `normalize=True`.
        """
        plt.imshow(cm, interpolation='nearest', cmap=cmap)
        plt.title(title)
        plt.colorbar()
        tick_marks = np.arange(len(classes))
        plt.xticks(tick_marks, classes, rotation=45)
        plt.yticks(tick_marks, classes)

        if normalize:
            cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
            print("Normalized confusion matrix")
        else:
            print('Confusion matrix, without normalization')

        # print(cm)

        thresh = cm.max() / 2.
        for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
            plt.text(j, i, cm[i, j],
                     horizontalalignment="center",
                     color="white" if cm[i, j] > thresh else "black")

        plt.tight_layout()
        plt.ylabel('True label')
        plt.xlabel('Predicted label')
