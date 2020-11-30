import os
import h5py
import skvideo.io
from PIL import Image

def get_frames():
    info_path = 'VSFA_LIVE-Qualcomminfo.mat'
    video_folder = r'D:\datasets\VQA\LIVE-Qualcomm'
    frame_folder = r'D:\datasets\VQAFrames\LIVEQualcomm'
    Info = h5py.File(info_path, 'r')
    video_names = [Info[Info['video_names'][0, :][i]][()].tobytes()[::2].decode() for i in
                   range(len(Info['video_names'][0, :]))]
    video_format = Info['video_format'][()].tobytes()[::2].decode()
    width = int(Info['width'][0])
    height = int(Info['height'][0])
    
    for video_name in video_names:
        # video_name: Artifacts/0723_ManUnderTree_GS5_03_20150723_130016.yuv
        # save_folder: D:\datasets\VQAFrames\LIVEQualcomm\0723_ManUnderTree_GS5_03_20150723_130016\
        save_folder = os.path.join(frame_folder, video_name.split('/')[1].split('.')[0])
        if not os.path.exists(save_folder):
            os.makedirs(save_folder)
        # video_data (ndarray): shape[lenght, H, W, C]
        video_data = skvideo.io.vread(os.path.join(video_folder, video_name), height, width, inputdict={'-pix_fmt':'yuvj420p'})
        video_length = video_data.shape[0]
        
        for frame_idx in range(video_length):
            frame = video_data[frame_idx]
            img = Image.fromarray(frame)
            save_name = os.path.join(save_folder, str(frame_idx)) + '.png'
            print(save_name)
            img.save(save_name)

def main():
    get_frames()

if __name__ == '__main__':
    main()