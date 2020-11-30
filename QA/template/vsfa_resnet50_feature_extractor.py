"""Extracting Content-Aware Perceptual Features using Pre-Trained ResNet-50"""
# Author: Dingquan Li
# Email: dingquanli AT pku DOT edu DOT cn
# Date: 2018/3/27
#
# CUDA_VISIBLE_DEVICES=0 python CNNfeatures.py --database=KoNViD-1k --frame_batch_size=64
# CUDA_VISIBLE_DEVICES=1 python CNNfeatures.py --database=CVD2014 --frame_batch_size=32
# CUDA_VISIBLE_DEVICES=0 python CNNfeatures.py --database=LIVE-Qualcomm --frame_batch_size=8

import torch
from torchvision import transforms, models
import torch.nn as nn
from torch.utils.data import Dataset
import skvideo.io
from PIL import Image
import os
import h5py
import numpy as np
import pandas as pd


class VideoDataset(Dataset):
    """
    Test or Validation
    """
    def __init__(self, database_info):
        self.video_folder = database_info['video_folder']
        self.video_names = database_info['video_names']
        self.data_list = self._make_dataset()
        self.transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
        
    def _make_dataset(self):
        data_list = []
        for idx in range(len(self.video_names)):
            video_name = self.video_names[idx]
            frame_names = list(os.walk(os.path.join(self.video_folder, str(video_name))))[0]
            frame_paths = list(map(lambda x: os.path.join(frame_names[0], x),
                                              sorted(frame_names[2], key=lambda x: int(x.split('.')[0]))))
            data_list.append((str(video_name), frame_paths))
        return data_list
        
    def __getitem__(self, index):
        video_name, frame_paths = self.data_list[index]
        
        len_video = len(frame_paths)
        frames = []
        for i in range(len_video):
            img = Image.open(frame_paths[i])
            img = self.transform(img)
            frames.append(img)
        transformed_data = torch.zeros([len_video, *frames[0].shape])
        for i in range(len_video):
            transformed_data[i] = frames[i]
        return video_name, transformed_data

    def __len__(self):
        return len(self.video_names)


class ResNet50(torch.nn.Module):
    """Modified ResNet50 for feature extraction"""
    def __init__(self):
        super(ResNet50, self).__init__()
        self.features = nn.Sequential(*list(models.resnet50(pretrained=True).children())[:-2])
        for p in self.features.parameters():
            p.requires_grad = False

    def forward(self, x):
        # features@: 7->res5c
        for ii, model in enumerate(self.features):
            x = model(x)
            if ii == 7:
                features_mean = nn.functional.adaptive_avg_pool2d(x, 1)
                features_std = global_std_pool2d(x)
                return features_mean, features_std


def global_std_pool2d(x):
    """2D global standard variation pooling"""
    return torch.std(x.view(x.size()[0], x.size()[1], -1, 1),
                     dim=2, keepdim=True)


def get_features(video_data, frame_batch_size=64, device='cuda'):
    """feature extraction"""
    extractor = ResNet50().to(device)
    video_length = video_data.shape[0]
    frame_start = 0
    frame_end = frame_start + frame_batch_size
    output1 = torch.Tensor().to(device)
    output2 = torch.Tensor().to(device)
    extractor.eval()
    with torch.no_grad():
	    while frame_end < video_length:
	        batch = video_data[frame_start:frame_end].to(device)
	        features_mean, features_std = extractor(batch)
	        output1 = torch.cat((output1, features_mean), 0)
	        output2 = torch.cat((output2, features_std), 0)
	        frame_end += frame_batch_size
	        frame_start += frame_batch_size

	    last_batch = video_data[frame_start:video_length].to(device)
	    features_mean, features_std = extractor(last_batch)
	    output1 = torch.cat((output1, features_mean), 0)
	    output2 = torch.cat((output2, features_std), 0)
	    output = torch.cat((output1, output2), 1).squeeze()

    return output


def main(**kwargs):
    '''
    CUDA_VISIBLE_DEVICES=1 python vsfa_resnet50_feature_extractor.py main --features_folder=/home/kedeleshier/lzq/VQAFeatures/VSFA/CVD2014 --video_folder=/media/kedeleshier/lzq/datasets/VQAFrames/CVD2014 --database_info_path=data/CVD2014/CVD2014_info.csv
    CUDA_VISIBLE_DEVICES=1 python vsfa_resnet50_feature_extractor.py main --features_folder=/home/kedeleshier/lzq/VQAFeatures/VSFA/VQC --video_folder=/media/kedeleshier/lzq/datasets/VQAFrames/VQC --database_info_path=data/VQC/VQC.csv
    :param kwargs:
    :return:
    '''
    features_folder = kwargs['features_folder']
    video_folder = kwargs['video_folder']
    database_info_path = kwargs['database_info_path']
    device = torch.device('cuda')
    if not os.path.exists(features_folder):
        os.makedirs(features_folder)

    # database information
    df_info = pd.read_csv(database_info_path)
    video_names = df_info['video_name'].values
    database_info = {
        'video_folder': video_folder,
        'video_names': video_names,
    }

    dataset = VideoDataset(database_info)

    for i in range(len(dataset)):
        video_name, current_data = dataset[i]
        print('Video {}: video name {}: length {}'.format(i, video_name, len(current_data)))
        features = get_features(current_data, frame_batch_size=16)
        feature_path = os.path.join(features_folder, str(video_name)) + '.npy'
        np.save(feature_path, features.to('cpu').numpy())

if __name__ == "__main__":
    import fire
    fire.Fire()
