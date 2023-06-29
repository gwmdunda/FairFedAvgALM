from __future__ import print_function

import os
import numpy as np
import pandas as pd
import random
import pickle

import torch
import torch.utils.data as data
from PIL import Image


class DatasetWrapper(data.Dataset):
    def __init__(self, dataset, transform) -> None:
        self.dataset = dataset
        self.transform = transform
    
    def __getitem__(self, index):
        if len(self.dataset[index]) == 2:
            X, y = self.dataset[index]
            X = self.transform(X)
            return X, y
        else:
            X, y, z = self.dataset[index]
            if self.transform:
                X = self.transform(X)
            return X, y, z
            
    def __len__(self):
        return len(self.dataset)

class DatasetFromSubset(data.Dataset):
    def __init__(self, subset, indices, transform=None):
        self.subset = subset
        self.transform = transform

    def __getitem__(self, index):
        x, y = self.subset[index]
        if self.transform:
            x = self.transform(x)
        return x, y

    def __len__(self):
        return len(self.subset)


class CelebaDataset(data.Dataset):
    def __init__(self, csv_path, img_dir, transform=None, label_attr = 'Attractive', protected_attr = 'Male'):
        
        df = pd.read_csv(csv_path, index_col=0)
        self.img_dir = img_dir
        self.csv_path = csv_path
        self.img_names = df.index.values
        self.y = df[label_attr].values
        self.p = df[protected_attr].values
        self.transform = transform
        
    def __getitem__(self, index):
        img = Image.open(os.path.join(self.img_dir,
                      self.img_names[index]))

        if self.transform is not None:
            img = self.transform(img)
            
        label = self.y[index]
        protected = self.p[index]
        return img, label, protected
        
    def __len__(self):
        return self.y.shape[0]

class ImSituVerbGender(data.Dataset):
    def __init__(self, dataset_dir, annotation_dir, image_dir, split = 'train', transform = None, randomness=True):

        self.dataset_dir = os.path.join(dataset_dir, "imsitu")
        self.split = split
        self.image_dir = image_dir
        self.annotation_dir = annotation_dir
        self.transform = transform

        verb_id_map = pickle.load(open(os.path.join(self.dataset_dir, self.annotation_dir, 'verb_id.map'), 'rb'))
        self.verb2id = verb_id_map['verb2id']
        self.id2verb = verb_id_map['id2verb']

        self.ann_data = pickle.load(open(os.path.join(self.dataset_dir, annotation_dir, split+".data"), 'rb'))

        self.verb_ann = np.zeros((len(self.ann_data), len(self.verb2id)))
        self.gender_ann = np.zeros((len(self.ann_data), 2), dtype=int)

        for index, ann in enumerate(self.ann_data):
            self.verb_ann[index][ann['verb']] = 1
            self.gender_ann[index][ann['gender']] = 1

        #Gender Balancing ------------------------------
        if randomness:
            man_idxs = np.nonzero(self.gender_ann[:, 0])[0]
            woman_idxs = np.nonzero(self.gender_ann[:, 1])[0]
            random.shuffle(man_idxs)# only blackout box is available for imSitu
            random.shuffle(woman_idxs)
            min_len = 10000 if self.split == 'train' else 3000
            selected_idxs = list(man_idxs[:min_len]) + list(woman_idxs[:min_len])
        else:
            selected_idxs = np.arange(0, len(self.ann_data), 1)

        self.ann_data = [self.ann_data[idx] for idx in selected_idxs]
        self.verb_ann = np.take(self.verb_ann, selected_idxs, axis=0)
        self.gender_ann = np.take(self.gender_ann, selected_idxs, axis=0)

        self.image_ids = range(len(self.ann_data))
        #-------------------------------------------------

        print("man size : {} and woman size: {}".format(len(np.nonzero( \
                self.gender_ann[:, 0])[0]), len(np.nonzero(self.gender_ann[:, 1])[0])))

    def __getitem__(self, index):
        img = self.ann_data[index]
        image_name = img['image_name']
        image_path_ = os.path.join(self.dataset_dir, self.image_dir, image_name)

        img_ = Image.open(image_path_).convert('RGB')

        if self.transform is not None:
            img_ = self.transform(img_)

        return img_, torch.argmax(torch.Tensor(self.verb_ann[index])), \
                torch.argmax(torch.LongTensor(self.gender_ann[index]))

    def getGenderWeights(self):
        return (self.gender_ann == 0).sum(axis = 0) / (1e-15 + \
                (self.gender_ann.sum(axis = 0) + (self.gender_ann == 0).sum(axis = 0) ))

    def getVerbWeights(self):
        return (self.verb_ann == 0).sum(axis = 0) / (1e-15 + self.verb_ann.sum(axis = 0))

    def blackout_img(self, img_name, img):
        if 'agent' not in self.masks_ann[img_name]['bb']:
            return img # if mask is not available, return the original img

        bb = self.masks_ann[img_name]['bb']['agent']
        if -1 in bb:
            return img # if mask if not available, return the original img
        else:
            xmin, ymin, xmax, ymax = self.masks_ann[img_name]['bb']['agent']
            width = self.masks_ann[img_name]['width']
            height = self.masks_ann[img_name]['height']
            black_img = Image.fromarray(np.zeros((img.size[1], img.size[0])))
            mask = np.zeros((width, height))
            for i in range(xmin, xmax):
                for j in range(ymin, ymax):
                    mask[j][i] = 1
            img_mask = Image.fromarray(255 * (mask > 0).astype('uint8')).resize((img.size[0], img.size[1]), Image.ANTIALIAS)
            return Image.composite(black_img, img, img_mask)

    def blackout_face(self, img_name, img):

        try:
            vertices = self.faces[img_name]
        except:
            return img

        # vertices = self.faces[img_name]
        width = img.size[1]
        height = img.size[0]

        black_img = Image.fromarray(np.zeros((img.size[1], img.size[0])))
        mask = np.zeros((width, height))
        for poly in vertices:
            xmin, ymin = poly[0].strip('()').split(',')
            xmax, ymax = poly[2].strip('()').split(',')
            for i in range(int(xmin), int(xmax)):
                for j in range(int(ymin), int(ymax)):
                    mask[j][i] = 1
        img_mask = Image.fromarray(255 * (mask > 0).astype('uint8')).resize((img.size[0], \
                img.size[1]), Image.ANTIALIAS)

        return Image.composite(black_img, img, img_mask)

    def __len__(self):
        return len(self.ann_data)