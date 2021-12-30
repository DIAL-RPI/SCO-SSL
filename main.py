import os
import numpy as np
import time
from itertools import cycle
import torch
import torch.nn as nn
from torch import optim
from torch.utils import data
import math
from dataset import create_ucla_folds, Dataset, TSDataset
from model import shadow_aug, ShadowUNet
from loss import DiceLoss
from utils import resample_array, output2file, generate_transform
from metric import eval
from config import cfg

def initial_net(net):
    for m in net.modules():
        if isinstance(m, nn.Conv2d) or isinstance(m, nn.Conv3d):
            nn.init.xavier_normal_(m.weight)
            nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.BatchNorm2d) or isinstance(m, nn.BatchNorm3d):
            nn.init.constant_(m.weight, 1)
            nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.Linear):
            nn.init.xavier_normal_(m.weight)
            nn.init.constant_(m.bias, 0)

def initial_teacher(teacher, student):
    sd_tea = teacher.state_dict()
    sd_stu = student.state_dict()
    for key in sd_tea:
        sd_tea[key] = sd_stu[key]
    teacher.load_state_dict(sd_tea)

def update_teacher(teacher, student, alpha):
    sd_tea = teacher.state_dict()
    sd_stu = student.state_dict()
    for key in sd_tea:
        sd_tea[key] = alpha * sd_tea[key] + (1.0-alpha) * sd_stu[key]
    teacher.load_state_dict(sd_tea)

def train(cfg):
    # record starting time
    start_time = time.localtime()
    time_stamp = time.strftime("%Y%m%d%H%M%S", start_time)
    acc_time = 0
    
    # create folders and filenames for results storage
    store_dir = '{}/model_{}'.format(cfg['model_path'], time_stamp)
    loss_fn = '{}/loss.txt'.format(store_dir)
    log_fn = '{}/log.txt'.format(store_dir)
    val_result_path = '{}/results_val'.format(store_dir)
    os.makedirs(val_result_path, exist_ok=True)
    test_result_path = '{}/results_test'.format(store_dir)
    os.makedirs(test_result_path, exist_ok=True)
    best_model_fn = '{}/cp-epoch_{}.pth.tar'.format(store_dir, 1)

    # create data split according to cfg['fold_fraction'] in config.py
    folds, _ = create_ucla_folds(data_path=cfg['data_path_train'], fraction=cfg['fold_fraction'], exclude_case=cfg['exclude_case'])
    
    # divide training data into 'labeled samples' and 'unlabeled sampled' according to cfg['labeled_num'] in config.py
    labeled_case = []
    unlabeled_case = []
    labeled_fold = []
    unlabeled_fold = []
    for record in folds[0]:
        case_name = record[1].split('-')[0]
        if (case_name not in labeled_case) and (case_name not in unlabeled_case):
            if len(labeled_case) < cfg['labeled_num']:
                labeled_case.append(case_name)
            else:
                unlabeled_case.append(case_name)

        if case_name in labeled_case:
            labeled_fold.append([record[0], record[1], record[2], record[3], True])
        else:
            unlabeled_fold.append([record[0], record[1], record[2], record[3], False])

    # create training fold (unlabeled)
    d_train_ul = TSDataset(unlabeled_fold, rs_size=cfg['rs_size'], rs_spacing=cfg['rs_spacing'], rs_intensity=cfg['rs_intensity'], label_map=cfg['label_map'], cls_num=cfg['cls_num'], aug_data=True)
    dl_train_ul = data.DataLoader(dataset=d_train_ul, batch_size=cfg['batch_size']-cfg['labeled_sample'], shuffle=True, pin_memory=True, drop_last=True, num_workers=cfg['cpu_thread_unlabeled'])

    # create training fold (labeled)
    train_fold = labeled_fold
    d_train = TSDataset(train_fold, rs_size=cfg['rs_size'], rs_spacing=cfg['rs_spacing'], rs_intensity=cfg['rs_intensity'], label_map=cfg['label_map'], cls_num=cfg['cls_num'], aug_data=True)
    dl_train = data.DataLoader(dataset=d_train, batch_size=cfg['labeled_sample'], shuffle=True, pin_memory=True, drop_last=True, num_workers=cfg['cpu_thread'])
    
    # create validaion fold
    val_fold = folds[1]
    d_val = Dataset(val_fold, rs_size=cfg['rs_size'], rs_spacing=cfg['rs_spacing'], rs_intensity=cfg['rs_intensity'], label_map=cfg['label_map'], cls_num=cfg['cls_num'], aug_data=False, load_dist=False, center_aligned=False)
    dl_val = data.DataLoader(dataset=d_val, batch_size=cfg['test_batch_size'], shuffle=False, pin_memory=True, drop_last=False, num_workers=cfg['cpu_thread'])

    # create testing fold
    test_fold = folds[2]
    d_test = Dataset(test_fold, rs_size=cfg['rs_size'], rs_spacing=cfg['rs_spacing'], rs_intensity=cfg['rs_intensity'], label_map=cfg['label_map'], cls_num=cfg['cls_num'], aug_data=False, load_dist=False, center_aligned=False)
    dl_test = data.DataLoader(dataset=d_test, batch_size=cfg['test_batch_size'], shuffle=False, pin_memory=True, drop_last=False, num_workers=cfg['cpu_thread'])

    # creat student model
    stu_model = ShadowUNet(in_ch=1, base_ch=64)
    stu_net = nn.DataParallel(module=stu_model)
    stu_net.cuda()
    initial_net(stu_net)

    # creat teacher model
    tea_model = ShadowUNet(in_ch=1, base_ch=64)
    tea_net = nn.DataParallel(module=tea_model)
    tea_net.cuda()
    initial_teacher(tea_net, stu_net)

    dice_loss = DiceLoss()
    bce_loss = nn.BCELoss()
    optimizer = optim.SGD(stu_net.parameters(), lr=cfg['lr'], momentum=0.99)
    
    best_val_acc = 0.0
    start_epoch = 0

    # print log
    print()
    log_line = "Training settings:\nModel: {}\nModel parameters: {}\nTraining/Validation/Testing samples: {}/{}/{}\nUnlabeled samples: {}\nStart time: {}\nConfiguration:\n".format(
        stu_model.description(), sum(x.numel() for x in stu_net.parameters()), len(d_train), len(d_val), len(d_test), len(d_train_ul), 
        time.strftime("%Y-%m-%d %H:%M:%S", start_time))
    log_line += 'Labeled/unlabeled subjects: {}/{}\n'.format(len(labeled_case), len(unlabeled_case))
    for cfg_key in cfg:
        log_line += ' --- {}: {}\n'.format(cfg_key, cfg[cfg_key])
    print(log_line)
    with open(log_fn, 'a') as log_file:
        log_file.write(log_line)

    # main loop of training and validation
    # evaluation on validation set is performed after each training epoch
    for epoch_id in range(start_epoch, cfg['epoch_num'], 1):
        t0 = time.perf_counter()
        
        # training
        torch.enable_grad()
        stu_net.train()
        tea_net.eval()
        epoch_loss = np.zeros(cfg['cls_num'], dtype=np.float)
        epoch_loss_num = np.zeros(cfg['cls_num'], dtype=np.int64)
        lamda_con = cfg['max_lamda'] * math.exp(-5*math.pow(1.0-float(epoch_id)/cfg['max_lamda_epoch'], 2.0)) if epoch_id < cfg['max_lamda_epoch'] else cfg['max_lamda']
        batch_id = 0
        for batch, batch_ul in zip(dl_train, cycle(dl_train_ul)):
            
            tea_image = torch.cat([batch['tea_data'], batch_ul['tea_data']], dim=0)
            tea_trans = torch.cat([batch['tea_transform'], batch_ul['tea_transform']], dim=0)
            tea_size = torch.cat([batch['tea_size'], batch_ul['tea_size']], dim=0)
            tea_spacing = torch.cat([batch['tea_spacing'], batch_ul['tea_spacing']], dim=0)
            tea_origin = torch.cat([batch['tea_origin'], batch_ul['tea_origin']], dim=0)

            stu_image = torch.cat([batch['stu_data'], batch_ul['stu_data']], dim=0)
            stu_label = torch.cat([batch['stu_label'], batch_ul['stu_label']], dim=0)
            stu_trans = torch.cat([batch['stu_transform'], batch_ul['stu_transform']], dim=0)
            stu_size = torch.cat([batch['stu_size'], batch_ul['stu_size']], dim=0)
            stu_spacing = torch.cat([batch['stu_spacing'], batch_ul['stu_spacing']], dim=0)
            stu_origin = torch.cat([batch['stu_origin'], batch_ul['stu_origin']], dim=0)

            flag = torch.cat([batch['label_exist'], batch_ul['label_exist']], dim=0)
            casename = batch['case'] + batch_ul['case']

            N = len(tea_image)

            rnd_idx = torch.randperm(N)
            tea_image = tea_image[rnd_idx]
            tea_trans = tea_trans[rnd_idx]
            tea_size = tea_size[rnd_idx]
            tea_spacing = tea_spacing[rnd_idx]
            tea_origin = tea_origin[rnd_idx]

            stu_image = stu_image[rnd_idx]
            stu_label = stu_label[rnd_idx]
            stu_trans = stu_trans[rnd_idx]
            stu_size = stu_size[rnd_idx]
            stu_spacing = stu_spacing[rnd_idx]
            stu_origin = stu_origin[rnd_idx]

            flag = flag[rnd_idx]
            casename_resorted = []
            for rnd_idx_i in rnd_idx.numpy():
                casename_resorted.append(casename[rnd_idx_i])
            casename = casename_resorted
            
            tea_image = tea_image.cuda()

            tea_image, tea_shadow = shadow_aug(tea_image, cfg, order='ascending')

            tea_pred = tea_net(tea_image, tea_shadow)
            tea_pred = tea_pred.detach()

            ps_label = torch.zeros_like(stu_label)
            for c in range(cfg['cls_num']):
                for n in range(N):
                    ps_array = tea_pred[n,c*2+1,:].cpu().numpy()

                    t_inv = generate_transform(tea_trans[n,:].numpy(), inverse=True)
                    tmp_array = resample_array(ps_array.copy(), tea_size[n,:].numpy(), tea_spacing[n,:].numpy(), tea_origin[n,:].numpy(), 
                                                stu_size[n,:].numpy(), stu_spacing[n,:].numpy(), stu_origin[n,:].numpy(), 
                                                transform=t_inv, linear=True)

                    t = generate_transform(stu_trans[n,:].numpy(), inverse=False)
                    tmp_array = resample_array(tmp_array, stu_size[n,:].numpy(), stu_spacing[n,:].numpy(), stu_origin[n,:].numpy(), 
                                                stu_size[n,:].numpy(), stu_spacing[n,:].numpy(), stu_origin[n,:].numpy(), 
                                                transform=t, linear=True)

                    ps_label[n,1,:] = torch.from_numpy(tmp_array)
                    ps_label[n,0,:] = 1 - torch.from_numpy(tmp_array)

            stu_image = stu_image.cuda()
            stu_label = stu_label.cuda()
            ps_label = ps_label.cuda()
            
            stu_image, stu_shadow = shadow_aug(stu_image, cfg, order='descending')

            stu_pred = stu_net(stu_image, stu_shadow)

            print_line = 'Epoch {0:d}/{1:d} (train) --- Progress {2:5.2f}% (+{3:02d})'.format(
                epoch_id+1, cfg['epoch_num'], 100.0 * batch_id * cfg['labeled_sample'] / len(d_train), N)

            loss_sup = dice_loss(stu_pred, stu_label, flag)
            epoch_loss[0] += loss_sup.item()
            epoch_loss_num[0] += 1
            loss_con = bce_loss(stu_pred, ps_label) # consistency loss is also calculated on labeled data to mitigate overfitting on small-size labeled data
            loss = loss_sup + lamda_con * loss_con

            print_line += ' --- Loss: {0:.6f}/{1:.6f}/{2:.6f} - {3:.6f}'.format(
                loss.item(), 
                loss_sup.item(), 
                loss_con.item(), 
                lamda_con)
            print(print_line)
            
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            del tea_image, tea_shadow, stu_image, stu_label, stu_shadow, ps_label, stu_pred, tea_pred, loss, loss_sup, loss_con
            
            update_teacher(tea_net, stu_net, alpha=0.99)
            batch_id += 1

        train_loss=np.sum(epoch_loss)/np.sum(epoch_loss_num)
        epoch_loss = epoch_loss / epoch_loss_num

        print_line = 'Epoch {0:d}/{1:d} (train) --- Loss: {2:.6f} ({3:s})\n'.format(
            epoch_id+1, cfg['epoch_num'], train_loss, '/'.join(['%.6f']*len(epoch_loss)) % tuple(epoch_loss))
        print(print_line)

        # validation
        torch.no_grad()
        stu_net.eval()
        tea_net.eval()
        for batch_id, batch in enumerate(dl_val):
            image = batch['data']
            flag = batch['label_exist']
            casename = batch['case']
            N = len(image)

            image = image.cuda()

            tea_pred = tea_net(image)

            print_line = 'Epoch {0:d}/{1:d} (val) --- Progress {2:5.2f}% (+{3:d})'.format(
                epoch_id+1, cfg['epoch_num'], 100.0 * batch_id * cfg['test_batch_size'] / len(d_val), N)
            print(print_line)

            for c in range(cfg['cls_num']):
                tea_pred_bin = torch.argmax(tea_pred[:,c*2:c*2+2], dim=1, keepdim=True)
                for i in range(N):
                    if flag[i, c] > 0:
                        # output teacher prediction
                        tea_mask = tea_pred_bin[i,:].contiguous().cpu().numpy().copy().astype(dtype=np.uint8)
                        tea_mask = np.squeeze(tea_mask)
                        tea_mask = resample_array(
                                tea_mask, batch['size'][i].numpy(), batch['spacing'][i].numpy(), batch['origin'][i].numpy(), 
                                batch['org_size'][i].numpy(), batch['org_spacing'][i].numpy(), batch['org_origin'][i].numpy())
                        output2file(tea_mask, batch['org_size'][i].numpy(), batch['org_spacing'][i].numpy(), batch['org_origin'][i].numpy(), 
                            '{}/{}@{}@{}.nii.gz'.format(val_result_path, batch['dataset'][i], batch['case'][i], c+1))
                        

                del tea_pred_bin

            del image, tea_pred
        
        tea_dsc, tea_asd, tea_hd, tea_dsc_m, tea_asd_m, tea_hd_m = eval(
            pd_path=val_result_path, gt_entries=val_fold, label_map=cfg['label_map'], cls_num=cfg['cls_num'], 
            metric_fn='val_results-epoch-{0:04d}'.format(epoch_id+1), calc_asd=False)
        
        print_line = 'Epoch {0:d}/{1:d} (val - teacher) --- DSC {2:.2f} ({3:s})% --- ASD {4:.2f} ({5:s})mm --- HD {6:.2f} ({7:s})mm'.format(
            epoch_id+1, cfg['epoch_num'], 
            tea_dsc_m*100.0, '/'.join(['%.2f']*len(tea_dsc[:,0])) % tuple(tea_dsc[:,0]*100.0), 
            tea_asd_m, '/'.join(['%.2f']*len(tea_asd[:,0])) % tuple(tea_asd[:,0]),
            tea_hd_m, '/'.join(['%.2f']*len(tea_hd[:,0])) % tuple(tea_hd[:,0]))
        print(print_line)
        
        t1 = time.perf_counter()
        epoch_t = t1 - t0
        acc_time += epoch_t
        print("Epoch time cost: {h:>02d}:{m:>02d}:{s:>02d}\n".format(
            h=int(epoch_t) // 3600, m=(int(epoch_t) % 3600) // 60, s=int(epoch_t) % 60))

        loss_line = '{epoch:>05d}\t{train_loss:>8.6f}\t{class_loss:s}\t{tea_val_dsc:>8.6f}\t{tea_val_dsc_cls:s}\n'.format(
            epoch=epoch_id+1, train_loss=train_loss, class_loss='\t'.join(['%8.6f']*len(epoch_loss)) % tuple(epoch_loss), 
            tea_val_dsc=tea_dsc_m, tea_val_dsc_cls='\t'.join(['%8.6f']*len(tea_dsc[:,0])) % tuple(tea_dsc[:,0]), 
            )
        with open(loss_fn, 'a') as loss_file:
            loss_file.write(loss_line)

        # save best model
        if epoch_id == 0 or tea_dsc_m > best_val_acc:
            # remove former best model
            if os.path.exists(best_model_fn):
                os.remove(best_model_fn)
            # save current best model
            best_val_acc = tea_dsc_m
            best_model_fn = '{}/cp-epoch_{}.pth.tar'.format(store_dir, epoch_id+1)                            
            torch.save({
                        'epoch':epoch_id,
                        'acc_time':acc_time,
                        'time_stamp':time_stamp,
                        'best_val_acc':best_val_acc,
                        'best_model_filename':best_model_fn,
                        'teacher_model_state_dict':tea_net.state_dict(),
                        'model_state_dict':stu_net.state_dict(),
                        'optimizer_state_dict':optimizer.state_dict()}, 
                        best_model_fn)
            print('Best model (epoch = {}) saved.\n'.format(epoch_id+1))

    # print log
    with open(log_fn, 'a') as log_file:
        log_file.write("Finish time: {finish_time}\nTotal training time: {h:>02d}:{m:>02d}:{s:>02d}\n\n".format(
                finish_time=time.strftime("%Y-%m-%d %H:%M:%S", time.localtime()),
                h=int(acc_time) // 3600, m=(int(acc_time) % 3600) // 60, s=int(acc_time) % 60))
                    
    # test
    tea_net.load_state_dict(torch.load(best_model_fn)['teacher_model_state_dict'])
    torch.no_grad()
    tea_net.eval()
    for batch_id, batch in enumerate(dl_test):
        image = batch['data']
        n = len(image)

        image = image.cuda()

        pred = tea_net(image)

        print_line = 'Testing --- Progress {0:5.2f}% (+{1:d})'.format(100.0 * batch_id * cfg['test_batch_size'] / len(d_test), n)
        print(print_line)

        for c in range(cfg['cls_num']):
            pred_bin = torch.argmax(pred[:,c*2:c*2+2], dim=1, keepdim=True)
            for i in range(n):
                mask = pred_bin[i,:].contiguous().cpu().numpy().copy().astype(dtype=np.uint8)
                mask = np.squeeze(mask)
                mask = resample_array(
                    mask, batch['size'][i].numpy(), batch['spacing'][i].numpy(), batch['origin'][i].numpy(), 
                    batch['org_size'][i].numpy(), batch['org_spacing'][i].numpy(), batch['org_origin'][i].numpy())
                output2file(mask, batch['org_size'][i].numpy(), batch['org_spacing'][i].numpy(), batch['org_origin'][i].numpy(), 
                    '{}/{}@{}@{}.nii.gz'.format(test_result_path, batch['dataset'][i], batch['case'][i], c+1))
            del pred_bin
        
        del image, pred
    
    dsc, asd, hd, dsc_m, asd_m, hd_m = eval(
        pd_path=test_result_path, gt_entries=test_fold, label_map=cfg['label_map'], cls_num=cfg['cls_num'], 
        metric_fn='test_results', calc_asd=True, keep_largest=True)
    
    print_line = 'Testing fold --- DSC {0:.2f} ({1:s})% --- ASD {2:.2f} ({3:s})mm --- HD {4:.2f} ({5:s})mm'.format(
        dsc_m*100.0, '/'.join(['%.2f']*len(dsc[:,0])) % tuple(dsc[:,0]*100.0), 
        asd_m, '/'.join(['%.2f']*len(asd[:,0])) % tuple(asd[:,0]),
        hd_m, '/'.join(['%.2f']*len(hd[:,0])) % tuple(hd[:,0]))
    print(print_line)

if __name__ == '__main__':
    os.environ['CUDA_VISIBLE_DEVICES'] = cfg['gpu']

    train(cfg=cfg)