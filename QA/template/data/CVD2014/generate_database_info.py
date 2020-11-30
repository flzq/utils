import h5py
import numpy as np
import pandas as pd


def generate_database_info():
    info_path = 'CVD2014info.mat'
    data = h5py.File(info_path, 'r')
    video_names = data['video_names']
    scores = data['scores']
    # video_name = video_names[0][0]
    # obj = data[video_name]
    # str = ''.join(chr(i) for i in obj[:])
    # print(str)
    video_names_list = []
    scores_list = []
    for idx in range(video_names.shape[1]):
        video_name = video_names[0][idx]
        score = scores[0][idx]
        obj = data[video_name]
        
        # Test1/City/Test01_City_D01.avi
        name = ''.join(chr(i) for i in obj[:])
        # Test1/City/Test01_City_D01.avi --> Test1/City/Test01_City_D01
        name = name.split('.')[0]
        # Test1/City/Test01_City_D01 --> Test01_City_D01
        name = name.split('/')[-1]  # Test01_City_D01
        
        video_names_list.append(name)
        scores_list.append(score)
    database_info = np.array([video_names_list, scores_list]).T
    df_database_info = pd.DataFrame(database_info, columns=['video_name', 'MOS'])
    df_database_info.to_csv('CVD2014_info.csv')

def main():
    generate_database_info()

if __name__ == '__main__':
    main()
    