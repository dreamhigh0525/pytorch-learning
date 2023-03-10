import albumentations as A
from albumentations.pytorch.transforms import ToTensorV2
from config import Phase


## MEMO: Detection models will normalize the images internally as seen in
## https://github.com/pytorch/vision/blob/d2c763e14efe57e4bf3ebf916ec243ce8ce3315c/torchvision/models/detection/faster_rcnn.py#L227
def get_transforms(phase: Phase) -> A.Compose:
    if phase == Phase.TRIAN:
        transforms = A.Compose([
            A.LongestMaxSize(max_size=1333),
            A.HorizontalFlip(p=0.5),
            A.RandomBrightnessContrast(brightness_limit=0.2, contrast_limit=0.2, p=0.5),
            A.GaussNoise(var_limit=(5, 25), p=0.5),
            #A.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),  ## default mean and std
            ToTensorV2(p=1)
        ],
        p=1.0,
        bbox_params=A.BboxParams(
            format="pascal_voc", min_area=0, min_visibility=0, label_fields=["labels"]
        ))
    elif phase == Phase.VAL:
        transforms = A.Compose([
            A.LongestMaxSize(max_size=1333),
            #A.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),  ## default mean and std
            ToTensorV2(p=1)
        ],
        p=1.0,
        bbox_params=A.BboxParams(
            format="pascal_voc", min_area=0, min_visibility=0, label_fields=["labels"]
        ))
    else:  ## test
        transforms = A.Compose([
            A.Resize(height=512, width=512, p=1),
            #A.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),  ## default mean and std
            ToTensorV2(p=1)
        ])
    return transforms