import os
import random
import pickle

from decord import VideoReader, cpu

import torch
from torch.utils.data import Dataset
from torchvision import transforms
import torchvision.transforms.functional as F
import torchvision.transforms as T
from PIL import Image
import numpy as np
import decord
import utils

mean_std = [[0.485, 0.456, 0.406],[0.229, 0.224, 0.225]]

class ThreePairedCropsTransform:
    """
    x0 : original view (ori_transform)
    q,k : paired random crops with shared crop parameters
    """

    def __init__(
        self,
        ori_transform,
        hflip_p=0.5,
        size=(224, 224),
        scale=(0.5, 1.0),
        ratio=(3./4., 4./3.),
        interpolation=F.InterpolationMode.BICUBIC,
        shift_pix = 30,
    ):
        self.ori_transform = ori_transform
        self.hflip_p = hflip_p
        self.size = size
        self.scale = scale
        self.ratio = ratio
        self.interpolation = interpolation
        self.shift_px = shift_pix

    def paired_random_crop(self, img1, img2):
        i1, j1, h1, w1 = transforms.RandomResizedCrop.get_params(
            img1, scale=self.scale, ratio=self.ratio
        )

        img1 = F.resized_crop(
            img1, i1, j1,  h1, w1, size=self.size, interpolation=self.interpolation
        )

        i2, j2, h2, w2 = transforms.RandomResizedCrop.get_params(
            img2, scale=self.scale, ratio=self.ratio
        )

        img2 = F.resized_crop(
            img2, i2, j2, h2, w2, size=self.size, interpolation=self.interpolation
        )

        # 左右反転もペアで揃える
        if random.random() < self.hflip_p:
            img1 = F.hflip(img1)
            img2 = F.hflip(img2)

        img1 = F.to_tensor(img1)
        img2 = F.to_tensor(img2)

        return img1, img2

    def __call__(self, x):
        # x0 : original view
        #x0 = self.ori_transform(x)

        # q,k : synchronized strong crops
        q, k = self.paired_random_crop(x,x)

        return q, k, k

class PairedRandomResizedCropWT:
    def __init__(
        self,
        hflip_p=0.5,
        size=(224, 224),
        scale=(0.5, 1.0),
        ratio=(3./4., 4./3.),
        interpolation=F.InterpolationMode.BICUBIC
    ):
        self.hflip_p = hflip_p
        self.size = size
        self.scale = scale
        self.ratio = ratio
        self.interpolation = interpolation

    def __call__(self, np_RGB_img_1, np_RGB_img_2):
        # Convert numpy images to PIL Images
        pil_RGB_img_1 = F.to_pil_image(np_RGB_img_1)
        pil_RGB_img_2 = F.to_pil_image(np_RGB_img_2)

        i, j, h, w = transforms.RandomResizedCrop.get_params(
            pil_RGB_img_1, scale=self.scale, ratio=self.ratio
        )
        # Apply the crop on both images
        cropped_img_1 = F.resized_crop(pil_RGB_img_1,
                                       i, j, h, w,
                                       size=(h,w))
        cropped_img_2 = F.resized_crop(pil_RGB_img_2,
                                       i, j, h, w,
                                       size=(h,w))
        
        resized_cropped_img_1 = F.resize(cropped_img_1,224)
        resized_cropped_img_2 = F.resize(cropped_img_2,224)

        final_cropped_img_1 = F.center_crop(resized_cropped_img_1, (224, 224))
        final_cropped_img_2 = F.center_crop(resized_cropped_img_2, (224, 224))


        if random.random() < self.hflip_p:
            final_cropped_img_1 = F.hflip(final_cropped_img_1)
            final_cropped_img_2 = F.hflip(final_cropped_img_2)

        return final_cropped_img_1, final_cropped_img_2

class PairedRandomResizedCrop:
    def __init__(
        self,
        hflip_p=0.5,
        size=(224, 224),
        scale=(0.5, 1.0),
        ratio=(3./4., 4./3.),
        interpolation=F.InterpolationMode.BICUBIC
    ):
        self.hflip_p = hflip_p
        self.size = size
        self.scale = scale
        self.ratio = ratio
        self.interpolation = interpolation

    def __call__(self, np_RGB_img_1, np_RGB_img_2):
        # Convert numpy images to PIL Images
        pil_RGB_img_1 = F.to_pil_image(np_RGB_img_1)
        pil_RGB_img_2 = F.to_pil_image(np_RGB_img_2)

        i, j, h, w = transforms.RandomResizedCrop.get_params(
            pil_RGB_img_1, scale=self.scale, ratio=self.ratio
        )
        # Apply the crop on both images
        cropped_img_1 = F.resized_crop(pil_RGB_img_1,
                                       i, j, h, w,
                                       size=self.size,
                                       interpolation=self.interpolation)
        cropped_img_2 = F.resized_crop(pil_RGB_img_2,
                                       i, j, h, w,
                                       size=self.size,
                                       interpolation=self.interpolation)
        

        if random.random() < self.hflip_p:
            cropped_img_1 = F.hflip(cropped_img_1)
            cropped_img_2 = F.hflip(cropped_img_2)

        return cropped_img_1, cropped_img_2

class PairedKinetics(Dataset):
    def __init__(
        self,
        root,
        max_distance=48,
        repeated_sampling=2
    ):
        super().__init__()
        self.root = root
        with open(
            os.path.join(self.root, "labels", f"label_1.0.pickle"), "rb"
        ) as f:
            self.samples = pickle.load(f)

        self.transforms = PairedRandomResizedCrop()
        self.basic_transform = transforms.Compose(
            [transforms.ToTensor(),
             transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])]
        )

        self.max_distance = max_distance
        self.repeated_sampling = repeated_sampling

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, index):
        sample = os.path.join(self.root, self.samples[index][1])
        vr = VideoReader(sample, num_threads=1, ctx=cpu(0))
        src_images = []
        tgt_images = []
        for i in range(self.repeated_sampling):
            src_image, tgt_image = self.load_frames(vr)
            src_image, tgt_image = self.transform(src_image, tgt_image)
            src_images.append(src_image)
            tgt_images.append(tgt_image)
        src_images = torch.stack(src_images, dim=0)
        tgt_images = torch.stack(tgt_images, dim=0)
        return src_images, tgt_images, 0

    def load_frames(self, vr):
        # handle temporal segments
        seg_len = len(vr)
        least_frames_num = self.max_distance + 1
        if seg_len >= least_frames_num:
            idx_cur = random.randint(0, seg_len - least_frames_num)
            interval = random.randint(4, self.max_distance)
            idx_fut = idx_cur + interval
        else:
            indices = random.sample(range(seg_len), 2)
            indices.sort()
            idx_cur, idx_fut = indices
        frame_cur = vr[idx_cur].asnumpy()
        frame_fut = vr[idx_fut].asnumpy()

        return frame_cur, frame_fut

    def transform(self, src_image, tgt_image):
        src_image, tgt_image = self.transforms(src_image, tgt_image)
        src_image = self.basic_transform(src_image)
        tgt_image = self.basic_transform(tgt_image)
        return src_image, tgt_image

#This for WorldTours data
class PairedKineticsWT(Dataset):
    def __init__(
        self,
        root,
        max_distance=48,
        repeated_sampling=2
    ):
        super().__init__()
        self.root = root
        with open(
            os.path.join(self.root, f"walking_tours_indexed.pkl"), "rb"
        ) as f:
            self.samples = pickle.load(f)

        #self.v_decoder = DecordInit()
        #v_reader = self.v_decoder(self.root)
        #self.samples = len(v_reader)

        self.transforms = PairedRandomResizedCropWT()
        self.basic_transform = transforms.Compose(
            [transforms.ToTensor(),
             transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])]
        )

        self.max_distance = max_distance
        self.repeated_sampling = repeated_sampling

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, index):
        sample = os.path.join(self.root, self.samples[index][1])
        vr = VideoReader(sample, num_threads=1, ctx=cpu(0))
        src_images = []
        tgt_images = []
        for i in range(self.repeated_sampling):
            src_image, tgt_image = self.load_frames(vr)
            src_image, tgt_image = self.transform(src_image, tgt_image)
            src_images.append(src_image)
            tgt_images.append(tgt_image)
        src_images = torch.stack(src_images, dim=0)
        tgt_images = torch.stack(tgt_images, dim=0)
        return src_images, tgt_images, 0

    def load_frames(self, vr):
        # handle temporal segments
        seg_len = len(vr)
        least_frames_num = self.max_distance + 1
        if seg_len >= least_frames_num * 2:  # 2枚おきにするので長さを2倍チェック
            idx_cur = random.randint(0, seg_len // 2 - least_frames_num)
            interval = random.randint(4, self.max_distance)
            idx_fut = idx_cur + interval
            # 実フレームインデックスに変換（2倍）
            idx_cur *= 2
            idx_fut *= 2
        else:
            indices = random.sample(range(0, seg_len, 2), 2)  # 2枚おきに候補を取る
            indices.sort()
            idx_cur, idx_fut = indices

        frame_cur = vr[idx_cur].asnumpy()
        frame_fut = vr[idx_fut].asnumpy()

        return frame_cur, frame_fut

    def transform(self, src_image, tgt_image):
        src_image, tgt_image = self.transforms(src_image, tgt_image)
        src_image = self.basic_transform(src_image)
        tgt_image = self.basic_transform(tgt_image)
        return src_image, tgt_image

