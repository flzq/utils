import h5py
import numpy as np
import pandas as pd


def generate_database_info():
    info_path = 'VSFA_LIVE-Qualcomminfo.mat'
    Info = h5py.File(info_path, 'r')
    video_names = [Info[Info['video_names'][0, :][i]][()].tobytes()[::2].decode() for i in
                   range(len(Info['video_names'][0, :]))]
    scores = Info['scores'][0, :]
    video_format = Info['video_format'][()].tobytes()[::2].decode()
    width = int(Info['width'][0])
    height = int(Info['height'][0])
    
    video_names_list = []
    for idx in range(len(video_names)):
        video_names_list.append(video_names[idx].split('/')[1].split('.')[0])
    database_info = np.array([video_names_list, scores]).T
    df_database_info = pd.DataFrame(database_info, columns=['video_name', 'MOS'])
    df_database_info.to_csv('LIVEQualcomm.csv')


def main():
    generate_database_info()


if __name__ == '__main__':
    main()