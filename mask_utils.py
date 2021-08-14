import torch
from torch.utils.data import Dataset, DataLoader
from torchvision.datasets.vision import VisionDataset
import transforms as T
import glob
import os
from PIL import Image
import numpy as np
import random
from xml.etree import cElementTree

class ConvertMasktoCOCO(object):
    CLASSES = (
        "Unknown", "with_mask", "without_mask"
    )
    def __call__(self, image, label_path):
        # return image, target
        filename = label_path.split('\\')[-1]
        # filename = filename.split('.')[0]

        parsed_xml = cElementTree.parse(label_path)
        root = parsed_xml.getroot()

        boxes = []
        classes = []
        objects = root.findall("object")
        for object in objects:
            bndbox = object.find("bndbox")
            xmin,ymin,xmax,ymax = [int(bndbox.find(name).text) for name in ["xmin","ymin","xmax","ymax"]]
            classes.append(self.CLASSES.index(object.find("name").text))
            boxes.append([xmin,ymin,xmax,ymax])

        # filename = label_path.split('\\')[-1]
        # filename = filename.split('.')[0]
        #
        #
        # f = open(label_path, 'r')
        #
        # boxes = []
        # classes = []
        #
        # lines = f.readlines()
        #
        # for line in lines[1:]:
        #     line = line.strip()  # delete line feed
        #     line = line.split(" ")
        #
        #     try:
        #         classes.append(self.CLASSES.index(line[0]))
        #     except:
        #         print(line[0])
        #         print(self.CLASSES.index(line[0]))
        #         raise Exception("Hell what the")
        #
        #     # classes.append(self.CLASSES.index(line[0]))
        #     line = np.array(line[1:5], dtype="int64")
        #     # x, y, w, h
        #     bbox = [n - 1 for n in [line[0], line[1], line[0] + line[2], line[1] + line[3]]]
        #     boxes.append(bbox)
        # f.close()

        boxes = torch.as_tensor(boxes, dtype=torch.float32)
        classes = torch.as_tensor(classes)
        if boxes.shape[0] < 1:
            print(label_path)

        target = {}
        target["boxes"] = boxes
        target["labels"] = classes
        target['name'] = torch.tensor([ord(i) for i in list(filename)], dtype=torch.int8) #convert filename in int8
        return image, target

class MaskDetection(VisionDataset):
    def __init__(self, img_folder, image_set, transforms = None):
        print("-"*30+"\n","ExDarkDetection Init")
        print(img_folder, image_set, transforms)
        self.root = img_folder
        self.image_set = image_set
        self.all_images_path = glob.glob(os.path.join(img_folder,image_set,"images","*"))
        self.all_labels_path = glob.glob(os.path.join(img_folder,image_set,"labels","*"))

        self._transforms = transforms

    def __getitem__(self, idx):
        data_path = self.all_images_path[idx]
        label_path = self.all_labels_path[idx]

        img = Image.open(data_path)
        img = img.convert("RGB") # image BGR -> RGB convert

        if self._transforms is not None:
            img, target = self._transforms(img, label_path)
            
        return img, target
    
    def get_height_and_width(self, idx):
        data_path = self.all_images_path[idx]

        img = Image.open(data_path)
        return img.size

    def __len__(self):
        return len(self.all_images_path)

# get exdark dataset
def get_Mask(root, image_set, transforms):
    t = [ConvertMasktoCOCO()]
    print(t)
    if transforms is not None:
        t.append(transforms)
    transforms = T.Compose(t)

    if image_set == "train":
        print("Create Train Dataset Mask")
        dataset = MaskDetection(img_folder=root,image_set=image_set, transforms=transforms)
    else:
        print("Create Test Dataset Mask")
        dataset = MaskDetection(img_folder=root, image_set=image_set, transforms=transforms)
    print("----" * 25)

    return dataset