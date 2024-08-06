from .vision import VisionDataset
from PIL import Image
import os
import os.path
import random
import io
from collections.abc import Iterable
import pickle
from torchvision.datasets.utils import verify_str_arg, iterable_to_str

def get_index(ele):
    index=ele.split(".")[0]
    return int(index)

class Word(VisionDataset):
    def __init__(self, root_content, transform=None):
        super(Word, self).__init__(root_content, transform=transform)


        self.content_image1_list = []
        self.content_image2_list = []
        self.mask_image_list = []
        self.content_list = []
        img_list1 = os.listdir(os.path.join(root_content, 't_t2'))
        img_list1.sort(key=get_index)
        img_list2 = os.listdir(os.path.join(root_content, 't_t1'))
        img_list2.sort(key=get_index)
        img_list3 = os.listdir(os.path.join(root_content, 'mask'))
        img_list3.sort(key=get_index)

        for img_name in img_list1:
            img = Image.open(os.path.join(os.path.join(root_content, 't_t2'), img_name)).convert('RGB')
            img_tensor = transform(img)
            self.content_image1_list.append(img_tensor)
        for img_name in img_list2:
            img = Image.open(os.path.join(os.path.join(root_content, 't_t1'), img_name)).convert('RGB')
            img_tensor = transform(img)
            self.content_image2_list.append(img_tensor)
        for img_name in img_list3:
            img = Image.open(os.path.join(os.path.join(root_content, 'mask'), img_name)).convert('L')
            img_tensor = transform(img)
            self.mask_image_list.append(img_tensor)
        with open(os.path.join(root_content, 'text1.txt')) as f:
            data = f.readlines()
            for line in data:
                line = line.strip()
                self.content_list.append(line)

    def __getitem__(self, index):

        return self.content_image1_list[index], self.content_image2_list[index], self.mask_image_list[index], self.content_list[index]

    def __len__(self):
        return len(self.content_image1_list)


class Word_Test(VisionDataset):
    def __init__(self, root_content, transform=None):
        super(Word_Test, self).__init__(root_content, transform=transform)

        self.guide_image_list = []
        self.content_list = []

        for img_name in os.listdir(os.path.join(root_content, 'ICDAR2013_t')):
            img = Image.open(os.path.join(os.path.join(root_content, 'ICDAR2013_t'), img_name)).convert('RGB')
            img_tensor = transform(img)
            self.guide_image_list.append(img_tensor)
        # with open(os.path.join(root_content, 'text1.txt')) as f:
        #     data = f.readlines()
        #     for line in data:
        #         line = line.strip()
        #         self.content_list.append(line)
    def __getitem__(self, index):

        return self.guide_image_list[index]
        # return self.guide_image_list[index], self.content_list[index]

    def __len__(self):
        return len(self.guide_image_list)

class Word_Test1(VisionDataset):
    def __init__(self, root_content, transform=None):
        super(Word_Test1, self).__init__(root_content, transform=transform)

        self.guide_image_list = []
        self.content_list = []
        self.stand_image_list = []
        img_list1 = os.listdir(os.path.join(root_content, 't_t1'))
        img_list1.sort(key=get_index)
        img_list2 = os.listdir(os.path.join(root_content, 'i_t_mask'))
        img_list2.sort(key=get_index)
        for img_name in img_list1:
            img = Image.open(os.path.join(os.path.join(root_content, 't_t1'), img_name)).convert('RGB')
            img_tensor = transform(img)
            self.guide_image_list.append(img_tensor)
        for img_name in img_list2:
            # img = Image.open(os.path.join(os.path.join(root_content, 'i_t'), img_name)).convert('RGB')
            # img_tensor = transform(img)
            img_name.replace('jpg','png')
            self.stand_image_list.append(os.path.join(os.path.join(root_content, 'i_t_mask'), img_name))
        with open(os.path.join(root_content, 'text1.txt')) as f:
            data = f.readlines()
            for line in data:
                line = line.strip()
                self.content_list.append(line)
    def __getitem__(self, index):

        return self.guide_image_list[index], self.content_list[index],self.stand_image_list[index]

    def __len__(self):
        return len(self.guide_image_list)

class Word_Test2(VisionDataset):
    def __init__(self, root_content, transform=None):
        super(Word_Test2, self).__init__(root_content, transform=transform)

        self.guide_image_list = []
        self.content_list = []
        img_list1 = os.listdir(os.path.join(root_content, 't_t1'))
        img_list1.sort(key=get_index)
        for img_name in img_list1:
            img = Image.open(os.path.join(os.path.join(root_content, 't_t1'), img_name)).convert('RGB')
            img_tensor = transform(img)
            self.guide_image_list.append(img_tensor)
        with open(os.path.join(root_content, 'text1.txt')) as f:
            data = f.readlines()
            for line in data:
                line = line.strip()
                self.content_list.append(line)
    def __getitem__(self, index):

        return self.guide_image_list[index], self.content_list[index]

    def __len__(self):
        return len(self.guide_image_list)

class Word_Test4(VisionDataset):
    def __init__(self, root_content, transform=None):
        super(Word_Test4, self).__init__(root_content, transform=transform)

        self.guide_image_list = []
        self.content_list = []
        self.stand_image_list = []
        self.stand_image_list_mask = []
        img_list1 = os.listdir(os.path.join(root_content, 't_t1'))
        img_list1.sort(key=get_index)
        img_list2 = os.listdir(os.path.join(root_content, 'i_t'))
        img_list2.sort(key=get_index)
        img_list3 = os.listdir(os.path.join(root_content, 'i_t_mask'))  
        img_list3.sort(key=get_index)
        for img_name in img_list1:
            img = Image.open(os.path.join(os.path.join(root_content, 't_t1'), img_name)).convert('RGB')
            img_tensor = transform(img)
            self.guide_image_list.append(img_tensor)
        for img_name in img_list2:
            img_name.replace('jpg','png')
            self.stand_image_list.append(os.path.join(os.path.join(root_content, 'i_t'), img_name))
        for img_name in img_list3:
            img_name.replace('jpg','png')
            self.stand_image_list_mask.append(os.path.join(os.path.join(root_content, 'i_t_mask'), img_name))
        with open(os.path.join(root_content, 'text1.txt')) as f:
            data = f.readlines()
            for line in data:
                line = line.strip()
                self.content_list.append(line)
    def __getitem__(self, index):

        return self.guide_image_list[index], self.content_list[index],self.stand_image_list[index],self.stand_image_list_mask[index]

    def __len__(self):
        return len(self.guide_image_list)