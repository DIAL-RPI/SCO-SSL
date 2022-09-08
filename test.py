import os
import numpy as np
import torch
import torch.nn as nn
from torch.utils import data
from dataset import create_data_folds, Dataset
from model import ShadowUNet
from utils import resample_array, output2file
from metric import eval
from config import cfg

# find the stored model file name in 'dir' path
def get_best_model_name(dir):
    for fn in os.listdir(dir):
        if fn.startswith('cp-epoch_') and fn.endswith('.pth.tar'):
            return fn
    return ''

def test(cfg):    
    # create folders and filenames for results storage
    store_dir = '{}/{}'.format(cfg['model_path'], cfg['model_folder'])
    test_result_path = '{}/results_test'.format(store_dir)
    os.makedirs(test_result_path, exist_ok=True)
    best_model_fn = '{}/{}'.format(store_dir, get_best_model_name(store_dir))

    # create data split according to cfg['fold_fraction'] in config.py
    # e.g., cfg['fold_fraction'] = [575,115,460] means 575/115/460 samples for training/validation/testing, respectively
    folds, _ = create_data_folds(data_path=cfg['data_path_train'], fraction=cfg['fold_fraction'], exclude_case=cfg['exclude_case'])
    
    # create testing fold
    test_fold = folds[2]
    d_test = Dataset(test_fold, rs_size=cfg['rs_size'], rs_spacing=cfg['rs_spacing'], rs_intensity=cfg['rs_intensity'], label_map=cfg['label_map'], cls_num=cfg['cls_num'], aug_data=False, center_aligned=False)
    dl_test = data.DataLoader(dataset=d_test, batch_size=cfg['test_batch_size'], shuffle=False, pin_memory=True, drop_last=False, num_workers=cfg['cpu_thread'])

    # creat teacher model
    tea_model = ShadowUNet(in_ch=1, base_ch=64)
    tea_model.initialization()
    tea_net = nn.DataParallel(module=tea_model)
    tea_net.cuda()

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
                # output teacher prediction to mask files
                mask = pred_bin[i,:].contiguous().cpu().numpy().copy().astype(dtype=np.uint8)
                mask = np.squeeze(mask)
                mask = resample_array(
                    mask, batch['size'][i].numpy(), batch['spacing'][i].numpy(), batch['origin'][i].numpy(), 
                    batch['org_size'][i].numpy(), batch['org_spacing'][i].numpy(), batch['org_origin'][i].numpy())
                output2file(mask, batch['org_size'][i].numpy(), batch['org_spacing'][i].numpy(), batch['org_origin'][i].numpy(), 
                    '{}/{}@{}@{}.nii.gz'.format(test_result_path, batch['dataset'][i], batch['case'][i], c+1))
            del pred_bin
        
        del image, pred
    
    # evaluate segmentation results on testing set
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

    cfg['model_folder'] = 'model_20220901012345' # specify the folder name of the trained model

    test(cfg=cfg)