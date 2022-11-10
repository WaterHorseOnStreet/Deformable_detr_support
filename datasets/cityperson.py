# ------------------------------------------------------------------------
# Deformable DETR
# Copyright (c) 2020 SenseTime. All Rights Reserved.
# Licensed under the Apache License, Version 2.0 [see LICENSE for details]
# ------------------------------------------------------------------------
# Modified from DETR (https://github.com/facebookresearch/detr)
# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
# ------------------------------------------------------------------------

"""
COCO dataset which returns image_id for evaluation.

Mostly copy-paste from https://github.com/pytorch/vision/blob/13b35ff/references/detection/coco_utils.py
"""
from pathlib import Path

import torch
import torch.utils.data
from pycocotools import mask as coco_mask
import torchvision
from .torchvision_datasets import CocoDetection as TvCocoDetection
from util.misc import get_local_rank, get_local_size
import datasets.transforms as T
import os
import PIL.Image as Image
from PIL import ImageFilter
import numpy as np
import random
import torch.nn.functional as F
import random

here = os.getcwd()
class CocoDetection(TvCocoDetection):
    def __init__(self, img_folder, ann_file, transforms, atten_transform, return_masks, cache_mode=False, local_rank=0, local_size=1):
        super(CocoDetection, self).__init__(img_folder, ann_file,
                                            cache_mode=cache_mode, local_rank=local_rank, local_size=local_size)
        self.root_path = img_folder
        self._transforms = transforms
        self._atten_transform = atten_transform
        self.prepare = ConvertCocoPolysToMask(return_masks)
        
        self.person_pool = os.listdir(os.path.join(here,'car_pool'))
        self.person_pool = [p for p in self.person_pool if p.endswith('.png')]
        random.shuffle(self.person_pool)
        self.person_patches_num = len(self.person_pool)

        
        
        self.patch_transform = T.Compose([T.randomColor(),
                                          T.RandomHorizontalFlip(),
                                          T.RandomResize([64,64]),
                                          T.ToTensor(),
                                         T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])

        
    def bbox2attenmap(self,bboxs,img):
        w,h = img.size
        atten = Image.new('L',size=img.size)
        if len(bboxs) != 0:
            for bbox in bboxs:
                ratio = None   
                if (bbox[3] - bbox[1]) < 40:
                    ratio = 0.01
                elif (bbox[3] - bbox[1]) >= 40 and (bbox[3] - bbox[1]) < 160:
                    ratio = 0.01
                else:
                    ratio = 0.1
                bbox = [int(corr) for corr in bbox]
                patch = self.get_atten_patch_from_bbox(bbox,ratio)
                atten.paste(patch,bbox)

        rgbArray = np.zeros((h,w,3), 'uint8')
        rgbArray[..., 0] = np.array(atten)
        rgbArray[..., 1] = np.array(atten)
        rgbArray[..., 2] = np.array(atten)
        atten = Image.fromarray(rgbArray)
        return atten
    
    def get_patch_pool(self,patch_name):
        patch_pool = os.listdir(os.path.join(here,'./{}_pool'.format(patch_name)))
        patch_pool = [p for p in patch_pool if p.endswith('.png')]
        random.shuffle(patch_pool)
        return patch_pool, len(patch_pool)

    def get_atten_patch_from_bbox(self,bbox,ratio):
        size_x = bbox[2] - bbox[0]
        size_y = bbox[3] - bbox[1]
        
        patch = np.zeros((size_x,size_y))
        sigma = np.array([[(size_x*2)//ratio,0],[0,(size_y*2)//ratio]])


        for i in range(size_x):
            for j in range(size_y):
                vec = np.array([i - size_x//2, j - size_y//2])
                value = np.exp(-0.5*np.dot(vec.transpose(),np.dot(np.linalg.inv(sigma),vec)))
                patch[i,j] = value

        patch = patch / np.max(patch)
        # patch = np.exp(patch) / np.sum(np.exp(patch))

        img_patch = Image.fromarray((patch.transpose()*255).astype(np.uint8))

        return img_patch
    
    def generate_attenmaps(self,root):
        count = 0
        for idx in range(len(self.ids)):
            img, target, path = super(CocoDetection, self).__getitem__(idx)
            image_id = self.ids[idx]
            if len(target) > 0:
                if 'ignore' in target[0]:
                    target = [tar for tar in target if tar['ignore'] == 0]

            target = {'image_id': image_id, 'annotations': target}
            img, target = self.prepare(img, target)
            
            attenmap = self.bbox2attenmap(target['boxes'],img)
            attenmap = attenmap.filter(ImageFilter.GaussianBlur(radius = 10))
            
            save_path = path.split('.')[0] + '_atten_map.jpg'
            print(save_path)
            attenmap.save(os.path.join(root,save_path))
            
            count = count + 1
            if count % 100 == 0:
                print("have generated {} attenmaps".format(count))

    def __getitem__(self, idx):
        img, target, path = super(CocoDetection, self).__getitem__(idx)
        image_id = self.ids[idx]
        if len(target) > 0:
            if 'ignore' in target[0]:
                target = [tar for tar in target if tar['ignore'] == 0]

        target = {'image_id': image_id, 'annotations': target}
        img, target = self.prepare(img, target)

        target_no_use = target
        target_no_use4patch = target

        # attenmap = self.bbox2attenmap(target['boxes'],img)
        # attenmap = attenmap.filter(ImageFilter.GaussianBlur(radius = 10))

        suffix = path.split('.')[0] + '_atten_map_pred_car.jpg'
        atten_path = Path(os.path.join(self.root_path, suffix))
        # atten_path = Path(os.path.join(os.getcwd(),'../../data/cityscape/leftImg8bit',suffix))
        if not atten_path.exists():
            suffix = path.split('.')[0] + '_atten_map_car.jpg'
            atten_path = os.path.join(self.root_path, suffix)
            # atten_path = os.path.join(os.getcwd(),'../../data/cityscape/leftImg8bit/val',suffix)
        attenmap = Image.open(atten_path).convert('RGB')
        
        # add more patch pools regarding 5 classes
        patch_path = self.person_pool[idx%self.person_patches_num]
        patch = Image.open(os.path.join(here,'car_pool',patch_path)).convert('RGB')
        
        if self._transforms is not None:
            state = torch.get_rng_state()
            old_state = random.getstate()
            img, target = self._transforms(img, target)
            torch.set_rng_state(state)
            random.setstate(old_state)
            attenmap, target_no_use = self._atten_transform(attenmap, target_no_use)
            patch, target_no_use4patch = self.patch_transform(patch,target_no_use4patch)

        attenmap = attenmap[2,:,:]
        return {'image':img, 'target':target, 'atten_map':attenmap.unsqueeze(0), 'patch':patch,'path':path}


def convert_coco_poly_to_mask(segmentations, height, width):
    masks = []
    for polygons in segmentations:
        rles = coco_mask.frPyObjects(polygons, height, width)
        mask = coco_mask.decode(rles)
        if len(mask.shape) < 3:
            mask = mask[..., None]
        mask = torch.as_tensor(mask, dtype=torch.uint8)
        mask = mask.any(dim=2)
        masks.append(mask)
    if masks:
        masks = torch.stack(masks, dim=0)
    else:
        masks = torch.zeros((0, height, width), dtype=torch.uint8)
    return masks


class ConvertCocoPolysToMask(object):
    def __init__(self, return_masks=False):
        self.return_masks = return_masks

    def __call__(self, image, target):
        w, h = image.size

        image_id = target["image_id"]
        image_id = torch.tensor([image_id])

        anno = target["annotations"]

        anno = [obj for obj in anno if 'iscrowd' not in obj or obj['iscrowd'] == 0]

        boxes = [obj["bbox"] for obj in anno]
        # guard against no boxes via resizing
        boxes = torch.as_tensor(boxes, dtype=torch.float32).reshape(-1, 4)
        boxes[:, 2:] += boxes[:, :2]
        boxes[:, 0::2].clamp_(min=0, max=w)
        boxes[:, 1::2].clamp_(min=0, max=h)

        classes = [obj["category_id"] for obj in anno]
        classes = torch.tensor(classes, dtype=torch.int64)

        if self.return_masks:
            segmentations = [obj["segmentation"] for obj in anno]
            masks = convert_coco_poly_to_mask(segmentations, h, w)

        keypoints = None
        if anno and "keypoints" in anno[0]:
            keypoints = [obj["keypoints"] for obj in anno]
            keypoints = torch.as_tensor(keypoints, dtype=torch.float32)
            num_keypoints = keypoints.shape[0]
            if num_keypoints:
                keypoints = keypoints.view(num_keypoints, -1, 3)

        keep = (boxes[:, 3] > boxes[:, 1]) & (boxes[:, 2] > boxes[:, 0])
        boxes = boxes[keep]
        classes = classes[keep]
        if self.return_masks:
            masks = masks[keep]
        if keypoints is not None:
            keypoints = keypoints[keep]

        target = {}
        target["boxes"] = boxes
        target["labels"] = classes
        if self.return_masks:
            target["masks"] = masks
        target["image_id"] = image_id
        if keypoints is not None:
            target["keypoints"] = keypoints

        # for conversion to coco api
        # area = torch.tensor([obj["area"] for obj in anno])
        iscrowd = torch.tensor([obj["iscrowd"] if "iscrowd" in obj else 0 for obj in anno])
        # target["area"] = area[keep]
        target["iscrowd"] = iscrowd[keep]

        target["orig_size"] = torch.as_tensor([int(h), int(w)])
        target["size"] = torch.as_tensor([int(h), int(w)])

        return image, target


def make_coco_transforms(image_set,is_atten):

    normalize = []
    normalize.append(T.ToTensor())
    if not is_atten:
        normalize.append(T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]))

    normalize = T.Compose(normalize)

    scales = [480, 512, 544, 576, 608, 640, 672, 704, 736, 768, 800]

    if image_set == 'train':
        return T.Compose([
            T.randomColor(),
            T.RandomHorizontalFlip(),
            T.RandomSelect(
                T.RandomResize(scales, max_size=1333),
                T.Compose([
                    T.RandomResize([400, 500, 600]),
                    T.RandomSizeCrop(384, 600),
                    T.RandomResize(scales, max_size=1333),
                ])
            ),
            normalize,
        ])

    if image_set == 'val':
        return T.Compose([
             #T.RandomResize([800], max_size=1333),
            normalize,
        ])

    raise ValueError(f'unknown {image_set}')



def build_cityperson(image_set, args):
    here = os.getcwd()
    root = Path(os.path.join(here,args.cityperson_path))
    assert root.exists(), f'provided COCO path {root} does not exist'
    mode = 'instances'
    # PATHS = {
    #     "train": (root / "leftImg8bit/" , root / "cityscapes_ann" / 'train.json'),
    #     "val": (root / "leftImg8bit"/"val", root / "cityscapes_ann" / 'val_gt.json'),
    # }
    PATHS = {
        "train": (root / "leftImg8bit/leftImg8bit_trainvaltest" , root / "cityscapes_ann" / 'car_filtered_gtFine_train.json'),
        "val": (root, root / "cityscapes_ann" / 'car_filtered_gtFine_val.json'),
    }

    img_folder, ann_file = PATHS[image_set]
    dataset = CocoDetection(img_folder, ann_file, transforms=make_coco_transforms(image_set,False), atten_transform = make_coco_transforms(image_set,True), return_masks=args.masks,
                            cache_mode=args.cache_mode, local_rank=get_local_rank(), local_size=get_local_size())
    # code below to generate the groudtruth attention map, comment it when attention map has generated
    # if image_set == 'train':
    #     dataset.generate_attenmaps(os.path.join(here,args.cityperson_path,"leftImg8bit"))
    # if image_set == 'val':
    #     dataset.generate_attenmaps(os.path.join(here,args.cityperson_path,"leftImg8bit/val"))
    return dataset
