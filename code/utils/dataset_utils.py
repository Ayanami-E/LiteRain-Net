import os
import random
import copy
from PIL import Image
import numpy as np

from torch.utils.data import Dataset
from torchvision.transforms import ToPILImage, Compose, RandomCrop, ToTensor
import torch

from utils.image_utils import random_augmentation, crop_img
from utils.degradation_utils import Degradation

    
class TrainDataset(Dataset):
    def __init__(self, args):
        super().__init__()
        self.args = args
        self.data_ids = []
        self.toTensor = ToTensor()

        self._init_ids()

    def _init_ids(self):
        data = self.args.data_dir
        file_names = os.listdir(data)
        self.data_ids+= [data + id for id in file_names]
        random.shuffle(self.data_ids)
        num_data = len(self.data_ids)
        print("Total number of training data: {}".format(num_data))

    def _crop_patch(self, img_1, img_2):
        H = img_1.shape[0]
        W = img_1.shape[1]
        ind_H = random.randint(0, H - self.args.patch_size)
        ind_W = random.randint(0, W - self.args.patch_size)

        patch_1 = img_1[ind_H:ind_H + self.args.patch_size, ind_W:ind_W + self.args.patch_size]
        patch_2 = img_2[ind_H:ind_H + self.args.patch_size, ind_W:ind_W + self.args.patch_size]

        return patch_1, patch_2

    def _get_data_gt(self, data_name):
        gt_name = data_name.split("input")[0] + 'target/' + data_name.split('/')[-1]
        return gt_name
    
    def __getitem__(self, index):
        sample = self.data_ids[index]
        degrad_img = crop_img(np.array(Image.open(sample).convert('RGB')), base=16)
        clean_name = self._get_data_gt(sample)
        clean_img = crop_img(np.array(Image.open(clean_name).convert('RGB')), base=16)
        degrad_patch, clean_patch = random_augmentation(*self._crop_patch(degrad_img, clean_img))
        clean_patch = self.toTensor(clean_patch)
        degrad_patch = self.toTensor(degrad_patch)
        return degrad_patch, clean_patch
    
    def __len__(self):
        return len(self.data_ids)


class TestDataset(Dataset):
    def __init__(self, args):
        super().__init__()
        self.args = args
        self.data_ids = []
        self.toTensor = ToTensor()

        self._init_ids()

    def _init_ids(self):
        # 使用 os.path.join 进行路径拼接
        data = self.args.data_path
        if not os.path.exists(data):
            raise FileNotFoundError(f"Test data directory not found: {data}")
        file_names = os.listdir(data)
        self.data_ids += [os.path.join(data, file_name) for file_name in file_names]
        print(f"Loaded {len(self.data_ids)} test files from {data}")

    def _get_data_gt(self, data_name):
        # 使用 os.path.dirname 和 os.path.basename 操作路径
        dir_name = os.path.dirname(data_name).replace("input", "target")
        file_name = os.path.basename(data_name)
        gt_name = os.path.join(dir_name, file_name)
        if not os.path.exists(gt_name):
            raise FileNotFoundError(f"Ground truth file not found: {gt_name}")
        return gt_name

    def __getitem__(self, index):
        sample = self.data_ids[index]
        degrad_img = crop_img(np.array(Image.open(sample).convert('RGB')), base=16)
        clean_name = self._get_data_gt(sample)
        clean_img = crop_img(np.array(Image.open(clean_name).convert('RGB')), base=16)
        clean_img = self.toTensor(clean_img)
        degrad_img = self.toTensor(degrad_img)
        degrad_name = os.path.basename(sample)[:-4]
        return degrad_name, degrad_img, clean_img

    def __len__(self):
        return len(self.data_ids)
