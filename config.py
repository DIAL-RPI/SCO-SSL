cfg = {}

cfg['cls_num'] = 1 # number of foreground classes (for TRUS image prostate segmentation, we have only one foreground class, i.e., prostate.)
cfg['gpu'] = '0' # to use multiple gpu: cfg['gpu'] = '0,1,2,3'
cfg['fold_fraction'] = [575,115,460] # data split
cfg['epoch_num'] = 400 # total epoch number for training
cfg['batch_size'] = 36 # batch size for training
cfg['test_batch_size'] = 16 # batch size for testing
cfg['labeled_sample'] = 12 # number of labeled samples in each batch
cfg['max_lamda'] = 0.1 # the maximum value of consistency training weight
cfg['max_lamda_epoch'] = 200 # the epoch number when the consistency training weight reaches its maximum value
cfg['lr'] = 0.001 # learning rate
cfg['model_path'] = '/home/models' # the path to store trained models
cfg['rs_size'] = [96,64,96] # resample size: [x, y, z]
cfg['rs_spacing'] = [1.0,1.0,1.0] # resample spacing: [x, y, z]. non-positive value means adaptive spacing fit the physical size: rs_size * rs_spacing = origin_size * origin_spacing
cfg['rs_intensity'] = [0.0, 255.0] # rescale intensity from [min, max] to [0, 1].
cfg['cpu_thread'] = 4 # multi-thread for data loading. zero means single thread.
cfg['cpu_thread_unlabeled'] = 8 # multi-thread for data loading. zero means single thread.
cfg['shadow_threshold'] = 60.0 # threshold for shadow extraction (in the range of [0, 255], pixels with intensity lower than the threshold are considered as shadow)
cfg['labeled_num'] = 58 # 10% of 575 training samples are used as labeled samples
cfg['unlabeled_num'] = cfg['fold_fraction'][0] - cfg['labeled_num']

# list of dataset name and path
cfg['data_path_train'] = [
    ['prostate_ucla', '/home/data/prostate_ucla'],
]

# map labels of different datasets to a uniform label map
cfg['label_map'] = {
    #'prostate_315-3-fold':{1:1},
    'prostate_ucla':{1:1},
}

# exclude any samples in the form of '[dataset_name, case_name]'
cfg['exclude_case'] = [
]