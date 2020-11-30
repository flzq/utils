from data import VSFADataset
import models
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
from utils import compute_metrics, formated_print_results, formated_print_avg_results
import os
import fire
from config import DefaultConfig
import json
from tqdm import tqdm


def evaluate_accuracy(net, criterion, val_loader, scale):
    """

    :param net:
    :param criterion:
    :param val_loader: # 验证集数据加载设定每次循环返回一个视频的所有帧[batch_size==1, n_frames_batch, C, n_frames, H, W]
    :param device:
    :return:
    """
    y_predict = np.zeros(len(val_loader))
    y_label = np.zeros(len(val_loader))

    net.eval()
    with torch.no_grad():
        for i, (features, length, label) in enumerate(val_loader):
            features = features.float().cuda()
            y_label[i] = scale * label.cpu().item()
            
            outputs = net(features, length.float())
            
            y_predict[i] = scale * outputs.cpu().item()
            # print(y_predict[i], label.cpu().item())
        
        # 计算 PLCC、SROCC、RMSE、KROCC
    val_SROCC, val_KROCC, val_PLCC, val_RMSE = compute_metrics(y_predict, y_label)

    return val_PLCC, val_SROCC, val_RMSE, val_KROCC, y_predict, y_label


def train(opt: DefaultConfig, idx: int):
    torch.manual_seed(20200813)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    np.random.seed(20200813)
    torch.utils.backcompat.broadcast_warning.enabled = True

    '''
    database information
    train_val_test information
    '''
    # database info
    feature_folder = opt.feature_folder

    database_info_path = opt.database_info_path
    # read database info
    info = pd.read_csv(database_info_path)
    file_names = info['video_name'].values
    feature_file_names = np.array([str(k) + '.npy' for k in file_names])
    labels = info['MOS'].values
    scale = labels.max()
    database_info = {  # 'video_folder': video_folder,
                    'feature_folder': feature_folder,
                     'feature_file_names': feature_file_names,
                     'labels': labels,
                     'scale': scale,
                    }
    # train_val_test info
    split_idx_file_path = opt.split_idx_file_path[idx]
    # read split info(train/val/test)
    split_info = pd.read_excel(split_idx_file_path)
    idx_all = split_info.iloc[:, 0].values
    split_status = split_info['status'].values
    train_idx = idx_all[split_status == 'train']
    validation_idx = idx_all[split_status == 'validation']
    test_idx = idx_all[split_status == 'test']
    idx_info = {
                'train': train_idx,
                'val': validation_idx,
                'test': test_idx,
            }
    # dataloader
    train_dataset = VSFADataset(idx_info, database_info, state='train', feature_shuffle=opt.train_feature_shuffle,
                                max_len=opt.max_len, feat_dim=opt.feature_dim)
    train_loader = torch.utils.data.DataLoader(dataset=train_dataset, batch_size=opt.train_batchsize, shuffle=True,
                                               num_workers=opt.num_workers)
    validation_dataset = VSFADataset(idx_info, database_info, state='val', feature_shuffle=opt.val_feature_shuffle,
                                     max_len=opt.max_len, feat_dim=opt.feature_dim)
    validation_loader = torch.utils.data.DataLoader(dataset=validation_dataset, batch_size=opt.val_batchsize,
                                                    shuffle=True, num_workers=opt.num_workers)
    test_dataset = VSFADataset(idx_info, database_info, state='test', feature_shuffle=opt.test_feature_shuffle,
                               max_len=opt.max_len, feat_dim=opt.feature_dim)
    test_loader = torch.utils.data.DataLoader(dataset=test_dataset, batch_size=opt.test_batchsize,
                                              num_workers=opt.num_workers)
    
    '''
    train information
        hyper params
        train info
    '''
    # init model
    device = opt.device
    model = getattr(models, opt.model_name)(input_size=opt.feature_dim).to(device)
    # hyper params
    num_epochs = opt.num_epochs
    lr = opt.lr
    # train info
    criterion = nn.L1Loss()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=opt.lr_decay)
    # scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=args.decay_interval, gamma=args.decay_ratio)
    
    best_val_SROCC = -1
    best_epoch = 1
    best_trained_model_filename = opt.model_name + '_' + opt.database + '_' + opt.timestamp
    best_trained_model_folder = os.path.join(opt.checkpoints_folder, best_trained_model_filename)
    best_trained_model_path = os.path.join(best_trained_model_folder, best_trained_model_filename + '_' + '{0:02d}'.format(idx) + '.pth')
    if not os.path.exists(best_trained_model_folder):
        os.makedirs(best_trained_model_folder)

    pbar = tqdm(range(1, num_epochs+1))
    # for epoch in range(1, num_epochs+1):
    for epoch in pbar:
        pbar.set_description("Processing [{}/{}]".format(idx+1, opt.num_iters))
        # 训练
        model.train()
        train_loss_sum, batch_count = 0.0, 0.0
        for i, (features, length, label) in enumerate(train_loader):
            features = features.to(device).float()
            label = label.to(device).float()
            outputs = model(features, length.float())

            loss = criterion(outputs, label)
            optimizer.zero_grad()  #
            loss.backward()
            optimizer.step()
            
            train_loss_sum += loss.cpu().item()
            batch_count += 1
        train_loss = train_loss_sum / batch_count
        
        
        # 验证
        val_PLCC, val_SROCC, val_RMSE, val_KROCC, y_val_pred, y_val = \
            evaluate_accuracy(model, criterion, validation_loader, scale)
        
        # Update the model with the best val SROCC
        if val_SROCC > best_val_SROCC:
            best_val_SROCC = val_SROCC
            best_epoch = epoch
            torch.save(model.state_dict(), best_trained_model_path)
            val_metrics = {'SROCC': val_SROCC, 'KROCC': val_KROCC, 'PLCC': val_PLCC, 'RMSE': val_RMSE}
            formated_print_results(val_metrics, 'Best Val', best_epoch)
            
    # 测试
    model.load_state_dict(torch.load(best_trained_model_path))
    test_PLCC, test_SROCC, test_RMSE, test_KROCC, y_test_pred, y_test = \
        evaluate_accuracy(model, criterion, test_loader, scale)
    # save test results
    # test_result_filename = opt.model_name + '_' + opt.database + '_' + opt.timestamp + '.json'
    # test_result_path = os.path.join(opt.results_folder, test_result_filename)
    test_recod: dict = {'metrics': {'test_SROCC': test_SROCC, 'test_KROCC': test_KROCC,'test_PLCC': test_PLCC, 'test_RMSE': test_RMSE},
                        'y_test_pred': y_test_pred.tolist(), 'y_test': y_test.tolist()}
    # with open(test_result_path, 'w') as f:
    #     json.dump(test_recod, f)
    # opt.save_config()
    test_metrics = {'SROCC': test_SROCC, 'KROCC': test_KROCC, 'PLCC': test_PLCC, 'RMSE': test_RMSE}
    formated_print_results(test_metrics, 'Test', best_epoch)

    return test_recod

def main(**kwargs):
    opt = DefaultConfig()
    opt.parse(kwargs)

    assert opt.feature_folder is not None
    assert opt.database is not None
    assert opt.database_info_path is not None
    assert opt.split_idx_file_path is not None
    assert opt.train_feature_shuffle is not None
    assert opt.train_feature_shuffle == True or opt.train_feature_shuffle == False
    assert opt.val_feature_shuffle is not None
    assert opt.val_feature_shuffle == True or opt.val_feature_shuffle == False
    assert opt.test_feature_shuffle is not None
    assert opt.test_feature_shuffle == True or opt.test_feature_shuffle == False
    assert opt.model_name is not None
    assert type(opt.split_idx_file_path) is list
    assert len(opt.split_idx_file_path) == opt.num_iters

    test_recod = {}
    srocc = np.zeros(opt.num_iters, dtype=np.float)
    krocc = np.zeros(opt.num_iters, dtype=np.float)
    plcc = np.zeros(opt.num_iters, dtype=np.float)
    rmse = np.zeros(opt.num_iters, dtype=np.float)
    for idx in range(opt.num_iters):
        recod = train(opt, idx)
        srocc[idx] = recod['metrics']['test_SROCC']
        krocc[idx] = recod['metrics']['test_KROCC']
        plcc[idx] = recod['metrics']['test_PLCC']
        rmse[idx] = recod['metrics']['test_RMSE']
        test_recod[str(idx)] = recod
    test_recod['avg'] = {'SROCC': {'mean': srocc.mean(), 'std': srocc.std()},
                         'KROCC': {'mean': krocc.mean(), 'std': krocc.std()},
                         'PLCC': {'mean': plcc.mean(), 'std': plcc.std()},
                         'RMSE': {'mean': rmse.mean(), 'std': rmse.std()},
                         }
    formated_print_avg_results(test_recod, opt.num_iters)
    # save all test data
    test_result_filename = opt.model_name + '_' + opt.database + '_' + opt.timestamp + '.json'
    test_result_path = os.path.join(opt.results_folder, test_result_filename)
    with open(test_result_path, 'w') as f:
        json.dump(test_recod, f)
    # save avg data
    test_avg_result_filename = opt.model_name + '_' + opt.database + '_' + opt.timestamp + '.txt'
    test_avg_result_path = os.path.join(opt.results_folder, test_avg_result_filename)
    with open(test_avg_result_path, 'w') as f:
        avg_metrics = test_recod['avg']
        content = ''
        for k, v in avg_metrics.items():
            content += "{}\n\tmean: {}\n\tstd: {}\n--------------\n".format(k, v['mean'], v['std'])
        f.write(content)
    # save config data
    opt.save_config()
    
if __name__ == '__main__':
    fire.Fire()