import warnings
warnings.filterwarnings('ignore')
from torch.utils.data import Dataset
import glob
import os
from skimage.io import imread
from torch.utils.data import DataLoader
from albumentations.pytorch import ToTensorV2
from albumentations import HorizontalFlip, VerticalFlip, RandomRotate90, Normalize, RandomCrop, RandomScale, Resize
from albumentations import OneOf, Compose
from skimage.color import rgb2hsv, hsv2rgb
import cv2
import ever as er
import random 
from collections import OrderedDict
from ever.interface import ConfigurableMixin
from torch.utils.data import SequentialSampler
from torch.utils.data import RandomSampler
from ever.api.data import distributed, CrossValSamplerGenerator
import numpy as np
import logging
from PIL import Image

logger = logging.getLogger(__name__)

# COLOR_MAP = OrderedDict(
#     Background=(255, 255, 255),
#     Building=(255, 0, 0),
#     Road=(255, 255, 0),
#     Water=(0, 0, 255),
#     Barren=(159, 129, 183),
#     Forest=(0, 255, 0),
#     Agricultural=(255, 195, 128),
# )

COLOR_MAP = OrderedDict(
    No_data=(255, 255, 255),
    Cultivated_land=(255, 0, 0),
    Forest=(255, 255, 0),
    Grassland=(0, 0, 255),
    Shrubland=(159, 129, 183),
    Water=(0, 255, 0),
    Artificial_urface=(255, 195, 128),
    Bareland=(255, 0, 255),
)


# COLOR_MAP = OrderedDict(
#     Background=(255, 255, 255),
#     Tree=(255, 0, 0),
#     Bush=(255, 255, 0),
#     Grass=(0, 0, 255),
# )

# COLOR_MAP = OrderedDict(
#     Background=(255, 255, 255),
#     Tree=(255, 0, 0),
#     Grass=(0, 0, 255),
# )

# COLOR_MAP = OrderedDict(
#     Background=(0, 0, 0),
#     Vegetation=(255, 255, 255),
# )


# LABEL_MAP = OrderedDict(
#     Background=0,
#     Building=1,
#     Road=2,
#     Water=3,
#     Barren=4,
#     Forest=5,
#     Agricultural=6
# )





def reclassify(cls):
    new_cls = np.ones_like(cls, dtype=np.int64) * -1
    for idx, label in enumerate(LABEL_MAP.values()):
        new_cls = np.where(cls == idx, np.ones_like(cls)*label, new_cls)
    return new_cls



class LoveDA(Dataset):
    def __init__(self, image_dir, mask_dir, transforms=None):
        self.rgb_filepath_list = []
        self.cls_filepath_list= []
        if isinstance(image_dir, list):
            for img_dir_path, mask_dir_path in zip(image_dir, mask_dir):
                self.batch_generate(img_dir_path, mask_dir_path)

        else:
            self.batch_generate(image_dir, mask_dir)

        self.transforms = transforms
        self.image_dir = image_dir
        


    def batch_generate(self, image_dir, mask_dir):
        # rgb_filepath_list = glob.glob(os.path.join(image_dir, '*.npy'))
        rgb_filepath_list = glob.glob(os.path.join(image_dir, '*.tif'))
        rgb_filepath_list += glob.glob(os.path.join(image_dir, '*.png'))

        logger.info('Dataset images: %d' % len(rgb_filepath_list))
        rgb_filename_list = [os.path.split(fp)[-1] for fp in rgb_filepath_list]
        cls_filepath_list = []
        if mask_dir is not None:
            for fname in rgb_filename_list:
                cls_filepath_list.append(os.path.join(mask_dir, fname))
        self.rgb_filepath_list += rgb_filepath_list
        self.cls_filepath_list += cls_filepath_list

    def __getitem__(self, idx):
        
        def custom_transforms(image, mask):
            b = image
            hsvb = rgb2hsv(b)
            hsvb_old = hsvb[:, :, 0]
            hsvb_new = hsvb[:, :, 0]
            
            val_max, val_min = 0.42, 0.05
            # variation = (val_max - val_min)/2
            # std_dev = variation/3
            std_dev = 0.5
            jitter = np.random.normal(0, std_dev)
            up, down = 0.42, 0.05

            if (jitter > 0):
                up, down = up-jitter, down
                hsvb_new = np.where(np.logical_and(hsvb_new<=val_max,hsvb_new>up), val_max, hsvb_new)
                hsvb_new = np.where(np.logical_and(hsvb_new<=up,hsvb_new>=down), hsvb_new+jitter, hsvb_new)
            
            if (jitter < 0):
                up, down = up, down-jitter
                hsvb_new = np.where(np.logical_and(hsvb_new>=val_min,hsvb_new<down), val_min, hsvb_new)
                hsvb_new = np.where(np.logical_and(hsvb_new>=down, hsvb_new<=up), hsvb_new+jitter, hsvb_new)
            
            mask_green = np.where(mask > 0, 1, 0)
            mask_no_green = np.where(mask == 0, 1, 0)
            
            hsvb_new = np.add(np.multiply(hsvb_new, mask_green), np.multiply(hsvb_old, mask_no_green))
            
            hsvb[:, :, 0] = hsvb_new
            
            norm_image = cv2.normalize(hsv2rgb(hsvb),None,alpha=0, beta=255, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_32F).astype(np.uint8)
            return norm_image
            
            
        image = imread(self.rgb_filepath_list[idx])
        if len(self.cls_filepath_list) > 0:
            mask = np.squeeze(np.load(self.cls_filepath_list[idx]+'.npy')).astype('float32')
            mask[mask == 2] = 1
            mask[mask == 3] = 1
            
      
            if self.transforms is not None:
                if (random.uniform(0, 1) > 0.5):
                    image = custom_transforms(image, mask)
                blob = self.transforms(image=image, mask=mask)
                image = blob['image']
                mask = blob['mask']

            return image, dict(cls=mask, fname=os.path.basename(self.rgb_filepath_list[idx]))
        else:
            if self.transforms is not None:
                # if (image.shape[2] != 3):
                #     print(os.path.basename(self.rgb_filepath_list[idx]))
                #     print(image.shape)
                #     os.remove(self.rgb_filepath_list[idx])
                blob = self.transforms(image=image, mask=mask)
                image = blob['image']

            return image, dict(fname=os.path.basename(self.rgb_filepath_list[idx]))

    def __len__(self):
        return len(self.rgb_filepath_list)


@er.registry.DATALOADER.register()
class LoveDALoader(DataLoader, ConfigurableMixin):
    def __init__(self, config):
        ConfigurableMixin.__init__(self, config)
        dataset = LoveDA(self.config.image_dir, self.config.mask_dir, self.config.transforms)
        if self.config.CV.i != -1:
            CV = CrossValSamplerGenerator(dataset, distributed=True, seed=2333)
            sampler_pairs = CV.k_fold(self.config.CV.k)
            train_sampler, val_sampler = sampler_pairs[self.config.CV.i]
            if self.config.training:
                sampler = train_sampler
            else:
                sampler = val_sampler
        else:
            sampler = RandomSampler(dataset) if self.config.training else SequentialSampler(
                dataset)

        super(LoveDALoader, self).__init__(dataset,
                                       self.config.batch_size,
                                       sampler=sampler,
                                       num_workers=self.config.num_workers,
                                       pin_memory=True,
                                       drop_last=True
                                       )
    def set_default_config(self):
        self.config.update(dict(
            image_dir=None,
            mask_dir=None,
            batch_size=16,
            num_workers=8,
            transforms=Compose([
                OneOf([
                    HorizontalFlip(True),
                    VerticalFlip(True),
                    RandomRotate90(True),
                ], p=0.75),
                Normalize(mean=(), std=(), max_pixel_value=1, always_apply=True),
                ToTensorV2()
            ]),
        ))

