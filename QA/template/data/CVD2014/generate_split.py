import pandas as pd
import numpy as np


def generate_split(idx):
    train_ratio = 0.6
    val_ratio = 0.2
    test_ratio = 0.2
    database_info_file_path = 'CVD2014_info.csv'
    split_file = 'train_val_test_split_{}.xlsx'.format(idx)
    print(split_file)
    
    df_info = pd.read_csv(database_info_file_path)
    file_names = df_info['video_name']
    num_videos = len(file_names)
    num_train_videos = int(train_ratio * num_videos)
    num_val_videos = int(val_ratio * num_videos)
    num_test_videos = num_videos - num_train_videos - num_val_videos
    status = np.array(['train'] * num_train_videos + ['validation'] * num_val_videos + ['test'] * num_test_videos)
    np.random.shuffle(status)
    
    split_info = np.array([file_names, status]).T
    df_split_info = pd.DataFrame(split_info, columns=['video_name', 'status'])
    # print(df_split_info.head())
    df_split_info.to_excel(split_file)


def main():
    for idx in range(0, 10):
        generate_split(idx)


if __name__ == '__main__':
    main()