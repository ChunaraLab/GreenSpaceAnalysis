from albumentations import Compose, OneOf, Normalize
from albumentations import HorizontalFlip, VerticalFlip, RandomRotate90, RandomCrop, GaussNoise
import albumentations as A
import ever as er

data = dict(
    train=dict(
        type='LoveDALoader',
        params=dict(
            image_dir=[
                # './LoveDA/Train/Urban/images_png/',
                # './LoveDA/Train/Rural/images_png/',
                # '/content/drive/MyDrive/green-spaces/Images/'
                '/scratch/mz2466/GREENSPACE/620Images/'
            ],
            mask_dir=[
                # './LoveDA/Train/Urban/masks_png/',
                # './LoveDA/Train/Rural/masks_png/',
                # '/content/drive/MyDrive/green-spaces/labels/'
                '/scratch/mz2466/GREENSPACE/620Labels/'
            ],
            
            transforms=Compose([
                RandomCrop(512, 512),
                OneOf([
                    HorizontalFlip(True),
                    VerticalFlip(True),
                    RandomRotate90(True)
                ], p=0.75),
                #GaussNoise(),
                #A.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1, p=1.0),
                Normalize(mean=(135.03, 128.968, 117.328),
                          std=(62.863, 57.524, 54.441),
                          max_pixel_value=1, always_apply=True),
                er.preprocess.albu.ToTensor()

            ]),
            CV=dict(k=10, i=-1),
            training=True,
            batch_size=16,
            num_workers=8,
        ),
    ),
    test=dict(
        type='LoveDALoader',
        params=dict(
            image_dir=[
                '/scratch/mz2466/GREENSPACE/New_Folder2'
            ],
            mask_dir=[
                '/scratch/mz2466/GREENSPACE/New_Folder_dummy_label/'
            ],
            transforms=Compose([
                Normalize(mean=(133.31, 127.336, 114.62),
                          std=(61.188, 55.28, 52.556),
                          max_pixel_value=1, always_apply=True),
                er.preprocess.albu.ToTensor()

            ]),
            CV=dict(k=10, i=-1),
            training=False,
            batch_size=4,
            num_workers=8,
        ),
    ),
)
optimizer = dict(
    type='sgd',
    params=dict(
        momentum=0.9,
        weight_decay=0.0001
    ),
    grad_clip=dict(
        max_norm=35,
        norm_type=2,
    )
)
learning_rate = dict(
    type='poly',
    params=dict(
        base_lr=0.005,
        power=0.9,
        max_iters=4000,
    ))
train = dict(
    forward_times=1,
    num_iters=4000,
    eval_per_epoch=True,
    summary_grads=False,
    summary_weights=False,
    distributed=True,
    apex_sync_bn=True,
    sync_bn=True,
    eval_after_train=True,
    log_interval_step=50,
    save_ckpt_interval_epoch=1000,
    eval_interval_epoch=20,
)

test = dict(

)

