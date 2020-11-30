import warnings
import torch
import time
import os


class DefaultConfig(object):
    """
        
        '''
        experience id
        '''
        exp_id = None  # e.g.: 0/1/2(type: int)
        
        '''
        model information
        '''
        model_name = None
        use_gpu = True
        
        '''
        database information
        '''
        # database info
        database = None
        video_folder = None  # save videos frames
        feature_folder = None
        database_info_path = None
        split_idx_file_path = None
        max_len = 400  # 每个视频的最大长度
        feature_dim = 4096  # 每帧得到的特征长度
        train_feature_shuffle = None  # 每个视频的特征是否要shuffle
        val_feature_shuffle = None
        test_feature_shuffle = None
        # dataset & dataloader info
        train_batchsize = 16
        val_batchsize = 1
        test_batchsize = 1
        num_workers = 1
        
        '''
        train information
        '''
        # hypter params
        num_epochs = 20
        lr = 1e-4
        lr_decay = 0.95
    """
    def __init__(self):
        '''
        other information
        '''
        # 格式化时间戳，用于实验记录
        self.timestamp = time.strftime('%Y%m%d%H%M%S')
        # 实验迭代次数
        self.num_iters: int = 10
        
        '''
        model information
        '''
        self.model_name = None
        self.use_gpu = True
        self.device = None
        
        '''
        database information
        '''
        # database info
        self.database: str = None
        self.feature_folder: str = None
        self.database_info_path: str = None
        self.split_idx_file_path: str = None
        '''
        KonVid max len: 240
        CVD2014 max len: 830
        VQC max len: 1202
        LIVEQualcomm max len: 526
        '''
        self.max_len: int = 2000  # 每个视频的最大长度
        self.feature_dim: int = 4096  # 每帧得到的特征长度
        self.train_feature_shuffle: bool = None  # 每个视频的特征是否要shuffle
        self.val_feature_shuffle: bool = None
        self.test_feature_shuffle: bool = None
        # dataset & dataloader info
        self.train_batchsize: int = 16
        self.val_batchsize: int = 1
        self.test_batchsize: int = 1
        self.num_workers: int = 1
        
        '''
        train information
        '''
        # hypter params
        self.num_epochs: int = 2000
        self.lr: float = 1e-5
        self.lr_decay: float = 0.0
        
        '''
        保存结果、数据
        '''
        # 保存模型参数文件夹
        self.checkpoints_folder = './checkpoints'
        if not os.path.exists(self.checkpoints_folder):
            os.makedirs(self.checkpoints_folder)
        # 保存配置文件夹
        self.results_folder = './results'
        if not os.path.exists(self.results_folder):
            os.makedirs(self.results_folder)
    
    def save_config(self):
        # config_filename: VSFA_Konvid_20201103123033
        config_filename = self.model_name + '_' + self.database + '_' + self.timestamp + '.config'
        config_filename = os.path.join(self.results_folder, config_filename)
        config = 'user config:\n'
        for k, v in self.__dict__.items():
            if not k.startswith('_'):
                config += '\t' + k + ' : ' + str(getattr(self, k)) + '\n'
        with open(config_filename, 'w') as f:
            f.write(config)
        
    
    def parse(self, kwargs):
        for k, v in kwargs.items():
            assert hasattr(self, k), ("Warning: opt has not attribut %s" % k)
            setattr(self, k, v)
        
        self.device = torch.device('cuda') if self.use_gpu and torch.cuda.is_available() else torch.device('cpu')
        
        # print('user config: ')
        # for k, v in self.__dict__.items():
        #     if not k.startswith('_'):
        #         print('\t', k, ':', getattr(self, k))


def main(**kwargs):
    opt = DefaultConfig()
    opt.parse(kwargs)
    opt.save_config()


if __name__ == '__main__':
    import fire
    fire.Fire()