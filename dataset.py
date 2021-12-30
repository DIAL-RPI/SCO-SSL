import os
import sys
import torch
from torch.utils import data
import numpy as np
import random
import itk
import SimpleITK as sitk

def read_image(fname, imtype):
    reader = itk.ImageFileReader[imtype].New()
    reader.SetFileName(fname)
    reader.Update()
    image = reader.GetOutput()
    return image

def get_center_of_mass(label):
    arr = itk.GetArrayFromImage(label)
    mask = np.zeros_like(arr)
    mask[arr > 0] = 1
    inds = np.nonzero(mask)
    origin = np.array(label.GetOrigin())
    spacing = np.array(label.GetSpacing())
    cm = np.zeros_like(origin)
    cm[2] = np.mean(inds[0])
    cm[1] = np.mean(inds[1])
    cm[0] = np.mean(inds[2])
    cm = origin + cm * spacing
    return cm

def scan_path(d_name, d_path):
    entries = []
    if d_name == 'prostate_ucla':
        for case_name in os.listdir('{}/image'.format(d_path)):
            if case_name.startswith('Case'):
                case_id = int(case_name.split('Case')[1])
                for fn in os.listdir('{}/image/{}'.format(d_path, case_name)):
                    if fn.startswith('us_') and fn.endswith('.nii.gz'):
                        image_name = '{0:s}/image/{1:s}/{2:s}'.format(d_path, case_name, fn)
                        label_name = '{0:s}/label/{1:s}/{2:s}'.format(d_path, case_name, fn)
                        if os.path.isfile(image_name) and os.path.isfile(label_name):
                            entries.append([d_name, case_name, image_name, label_name, True])
    return entries

def create_ucla_folds(data_path, fraction, exclude_case):
    fold_file_name = '{0:s}/CV_UCLA-fold.txt'.format(sys.path[0])
    folds = {}
    if os.path.exists(fold_file_name):
        with open(fold_file_name, 'r') as fold_file:
            strlines = fold_file.readlines()
            for strline in strlines:
                strline = strline.rstrip('\n')
                params = strline.split()
                fold_id = int(params[0])
                if fold_id not in folds:
                    folds[fold_id] = []
                folds[fold_id].append([params[1], params[2], params[3], params[4], bool(params[5])])
    else:
        entries = []
        for [d_name, d_path] in data_path:
            entries.extend(scan_path(d_name, d_path))
        for e in entries:
            if e[0:2] in exclude_case:
                entries.remove(e)
        unique_cases = []
        for e in entries:
            if e[0:2] not in unique_cases:
                unique_cases.append(e[0:2])
        case_num = len(unique_cases)
        random.shuffle(unique_cases)
        ptr = 0
        for fold_id in range(len(fraction)):
            folds[fold_id] = []
        for fold_id in range(len(fraction)):
            fold_cases = unique_cases[ptr:ptr+fraction[fold_id]]
            for e in entries:
                if e[0:2] in fold_cases:
                    folds[fold_id].append(e)
            ptr += fraction[fold_id]

        with open(fold_file_name, 'w') as fold_file:
            for fold_id in range(len(fraction)):
                for i, [d_name, case_name, image_path, label_path, unlabeled] in enumerate(folds[fold_id]):
                    instance_id = int(image_path.split('/{}/us_'.format(case_name))[1].split('.nii.gz')[0])
                    instance_name = '{0:s}-{1:d}'.format(case_name, instance_id)
                    fold_file.write('{0:d} {1:s} {2:s} {3:s} {4:s} {5:s}\n'.format(fold_id, d_name, instance_name, image_path, label_path, str(unlabeled)))
                    folds[fold_id][i] = [d_name, instance_name, image_path, label_path, unlabeled]

    folds_size = [len(x) for x in folds.values()]

    return folds, folds_size

def normalize(x, min, max):
    factor = 1.0 / (max - min)
    x[x < min] = min
    x[x > max] = max
    x = (x - min) * factor
    return x

def generate_transform(rand):
    if rand:
        min_rotate = -0.05 # [rad]
        max_rotate = 0.05 # [rad]
        min_offset = -5.0 # [mm]
        max_offset = 5.0 # [mm]
        t = itk.Euler3DTransform[itk.D].New()
        euler_parameters = t.GetParameters()
        euler_parameters = itk.OptimizerParameters[itk.D](t.GetNumberOfParameters())
        offset_x = min_offset + random.random() * (max_offset - min_offset) # rotate
        offset_y = min_offset + random.random() * (max_offset - min_offset) # rotate
        offset_z = min_offset + random.random() * (max_offset - min_offset) # rotate
        rotate_x = min_rotate + random.random() * (max_rotate - min_rotate) # tranlate
        rotate_y = min_rotate + random.random() * (max_rotate - min_rotate) # tranlate
        rotate_z = min_rotate + random.random() * (max_rotate - min_rotate) # tranlate
        euler_parameters[0] = rotate_x # rotate
        euler_parameters[1] = rotate_y # rotate
        euler_parameters[2] = rotate_z # rotate
        euler_parameters[3] = offset_x # tranlate
        euler_parameters[4] = offset_y # tranlate
        euler_parameters[5] = offset_z # tranlate
        t.SetParameters(euler_parameters)
    else:
        offset_x = 0
        offset_y = 0
        offset_z = 0
        rotate_x = 0
        rotate_y = 0
        rotate_z = 0
        t = itk.IdentityTransform[itk.D, 3].New()
    return t, [offset_x, offset_y, offset_z, rotate_x, rotate_y, rotate_z]

def resample(image, imtype, size, spacing, origin, transform, linear, dtype, use_min_default):
    o_origin = image.GetOrigin()
    o_spacing = image.GetSpacing()
    o_size = image.GetBufferedRegion().GetSize()
    output = {}
    output['org_size'] = np.array(o_size, dtype=int)
    output['org_spacing'] = np.array(o_spacing, dtype=np.float32)
    output['org_origin'] = np.array(o_origin, dtype=np.float32)
    
    if origin is None: # if no origin point specified, center align the resampled image with the original image
        new_size = np.zeros(3, dtype=int)
        new_spacing = np.zeros(3, dtype=np.float32)
        new_origin = np.zeros(3, dtype=np.float32)
        for i in range(3):
            new_size[i] = size[i]
            if spacing[i] > 0:
                new_spacing[i] = spacing[i]
                new_origin[i] = o_origin[i] + o_size[i]*o_spacing[i]*0.5 - size[i]*spacing[i]*0.5
            else:
                new_spacing[i] = o_size[i] * o_spacing[i] / size[i]
                new_origin[i] = o_origin[i]
    else:
        new_size = np.array(size, dtype=int)
        new_spacing = np.array(spacing, dtype=np.float32)
        new_origin = np.array(origin, dtype=np.float32)

    output['size'] = new_size
    output['spacing'] = new_spacing
    output['origin'] = new_origin

    resampler = itk.ResampleImageFilter[imtype, imtype].New()
    resampler.SetInput(image)
    resampler.SetSize((int(new_size[0]), int(new_size[1]), int(new_size[2])))
    resampler.SetOutputSpacing((float(new_spacing[0]), float(new_spacing[1]), float(new_spacing[2])))
    resampler.SetOutputOrigin((float(new_origin[0]), float(new_origin[1]), float(new_origin[2])))
    resampler.SetTransform(transform)
    if linear:
        resampler.SetInterpolator(itk.LinearInterpolateImageFunction[imtype, itk.D].New())
    else:
        resampler.SetInterpolator(itk.NearestNeighborInterpolateImageFunction[imtype, itk.D].New())
    if use_min_default:
        resampler.SetDefaultPixelValue(int(np.min(itk.GetArrayFromImage(image))))
    else:
        resampler.SetDefaultPixelValue(int(np.max(itk.GetArrayFromImage(image))))
    resampler.Update()
    rs_image = resampler.GetOutput()
    image_array = itk.GetArrayFromImage(rs_image)
    image_array = image_array[np.newaxis, :].astype(dtype)
    output['array'] = image_array

    return output

def make_onehot(input, cls):
    oh = np.repeat(np.zeros_like(input), cls*2, axis=0)
    for i in range(cls):
        tmp = np.zeros_like(input)
        tmp[input==i+1] = 1
        oh[i*2+0,:] = 1-tmp
        oh[i*2+1,:] = tmp
    return oh

def make_flag(cls, labelmap):
    flag = np.zeros([cls, 1], dtype=np.float32)
    for key in labelmap:
        flag[labelmap[key]-1,0] = 1
    return flag

# dataset of 3D image volume
# 3D volumes are resampled from and center-aligned with the original images
class Dataset(data.Dataset):
    def __init__(self, ids, rs_size, rs_spacing, rs_intensity, label_map, cls_num, aug_data, load_dist, center_aligned):
        self.ImageType = itk.Image[itk.SS, 3]
        self.LabelType = itk.Image[itk.UC, 3]
        self.FloatType = itk.Image[itk.F, 3]
        self.ids = ids
        self.rs_size = rs_size
        self.rs_spacing = rs_spacing
        self.rs_intensity = rs_intensity
        self.label_map = label_map
        self.cls_num = cls_num
        self.aug_data = aug_data
        self.load_dist = load_dist
        self.center_aligned = center_aligned
        self.case_center = {}
    
    def __len__(self):
        return len(self.ids)

    def __getitem__(self, index):
        [d_name, casename, image_fn, label_fn, labeled] = self.ids[index]

        cm = None
        if self.center_aligned:
            if casename not in self.case_center:
                src_label = read_image(fname=label_fn, imtype=self.LabelType)
                c = get_center_of_mass(src_label)
                self.case_center[casename] = c - np.array(self.rs_size) * np.array(self.rs_spacing) * 0.5
            cm = self.case_center[casename]

        t, t_param = generate_transform(rand=self.aug_data)

        output = {}
        src_image = read_image(fname=image_fn, imtype=self.ImageType)
        image = resample(image=src_image, imtype=self.ImageType, size=self.rs_size, spacing=self.rs_spacing, origin=cm, 
                        transform=t, linear=True, dtype=np.float32, use_min_default=True)
        image['array'] = normalize(image['array'], min=self.rs_intensity[0], max=self.rs_intensity[1])
        
        if labeled:
            src_label = read_image(fname=label_fn, imtype=self.LabelType)
            label = resample(image=src_label, imtype=self.LabelType, size=self.rs_size, spacing=self.rs_spacing, origin=cm, 
                        transform=t, linear=False, dtype=np.int64, use_min_default=True)

            tmp_array = np.zeros_like(label['array'])
            lmap = self.label_map[d_name]
            for key in lmap:
                tmp_array[label['array'] == key] = lmap[key]
            label['array'] = tmp_array
            label_bin = make_onehot(label['array'], cls=self.cls_num)
            label_exist = make_flag(cls=self.cls_num, labelmap=self.label_map[d_name])
        else:
            label_bin = make_onehot(np.zeros_like(image['array'], dtype=np.int64), cls=self.cls_num)
            label_exist = np.zeros([self.cls_num, 1])

        output['data'] = torch.from_numpy(image['array'])
        output['label'] = torch.from_numpy(label_bin.astype(np.float32))
        output['label_exist'] = label_exist
        output['dataset'] = d_name
        output['case'] = casename
        output['size'] = image['size']
        output['spacing'] = image['spacing']
        output['origin'] = image['origin']
        output['transform'] = np.array(t_param, dtype=np.float32)
        output['org_size'] = image['org_size']
        output['org_spacing'] = image['org_spacing']
        output['org_origin'] = image['org_origin']
        output['eof'] = True

        if self.load_dist:
            if d_name == 'prostate_labeled':
                dist_fn = '{0:s}/uronav_data/Case{1:04d}/us_dist.nii.gz'.format(os.path.dirname(os.path.dirname(image_fn)), int(casename.split('Case')[1]))
                dist_polar_fn = '{0:s}/uronav_data/Case{1:04d}/us_dist_polar.nii.gz'.format(os.path.dirname(os.path.dirname(image_fn)), int(casename.split('Case')[1]))
            else:
                dist_fn = '{0:s}/us_dist.nii.gz'.format(os.path.dirname(label_fn))
                dist_polar_fn = '{0:s}/us_dist_polar.nii.gz'.format(os.path.dirname(label_fn))
            if labeled:
                src_dist = read_image(fname=dist_fn, imtype=self.FloatType)
                dist = resample(image=src_dist, imtype=self.FloatType, size=self.rs_size, spacing=self.rs_spacing, origin=cm, 
                                transform=t, linear=True, dtype=np.float32, use_min_default=True)
                #dist['array'] = np.repeat(dist['array'], 2, axis=0)
                #dist['array'][0,:] = 1 - dist['array'][1,:]
                dist_tensor = torch.from_numpy(dist['array'])

                src_dist_polar = read_image(fname=dist_polar_fn, imtype=self.FloatType)
                dist_polar = resample(image=src_dist_polar, imtype=self.FloatType, size=self.rs_size, spacing=self.rs_spacing, origin=cm, 
                                transform=t, linear=True, dtype=np.float32, use_min_default=True)
                dist_polar_tensor = torch.from_numpy(dist_polar['array'])
            else:
                #dist_array = np.repeat(np.zeros_like(image['array'], dtype=np.float32), 2, axis=0)
                dist_array = np.zeros_like(image['array'], dtype=np.float32)
                dist_tensor = torch.from_numpy(dist_array)
                dist_polar_array = np.zeros_like(image['array'], dtype=np.float32)
                dist_polar_tensor = torch.from_numpy(dist_polar_array)
            output['dist'] = dist_tensor
            output['dist_polar'] = dist_polar_tensor

        return output

def keep_largest_component(image, largest_n=1):
    arr = itk.GetArrayFromImage(image)
    c_filter = sitk.ConnectedComponentImageFilter()
    obj_arr = sitk.GetArrayFromImage(c_filter.Execute(sitk.GetImageFromArray(arr)))
    obj_num = c_filter.GetObjectCount()
    tmp_arr = np.zeros_like(obj_arr)

    if obj_num > 0:
        obj_vol = np.zeros(obj_num, dtype=np.int64)
        for obj_id in range(obj_num):
            tmp_arr = np.zeros_like(obj_arr)
            tmp_arr[obj_arr == obj_id+1] = 1
            obj_vol[obj_id] = np.sum(tmp_arr)

        sorted_obj_id = np.argsort(obj_vol)[::-1]
    
        for i in range(min(largest_n, obj_num)):
            tmp_arr[obj_arr == sorted_obj_id[i]+1] = 1
            
    output = itk.GetImageFromArray(tmp_arr.astype(np.int16))
    output.SetSpacing(image.GetSpacing())
    output.SetOrigin(image.GetOrigin())
    output.SetDirection(image.GetDirection())

    return output

# Dataset for Teacher-Student training manner
# Each sample will be augmented twice by different transforms
# One for teacher model, the other one for student model
class TSDataset(data.Dataset):
    def __init__(self, ids, rs_size, rs_spacing, rs_intensity, label_map, cls_num, aug_data):
        self.ImageType = itk.Image[itk.SS, 3]
        self.LabelType = itk.Image[itk.UC, 3]
        self.FloatType = itk.Image[itk.F, 3]
        self.ids = ids
        self.rs_size = rs_size
        self.rs_spacing = rs_spacing
        self.rs_intensity = rs_intensity
        self.label_map = label_map
        self.cls_num = cls_num
        self.aug_data = aug_data
    
    def __len__(self):
        return len(self.ids)

    def __getitem__(self, index):
        [d_name, casename, image_fn, label_fn, labeled] = self.ids[index]
        
        output = {}

        src_image = read_image(fname=image_fn, imtype=self.ImageType)
        if labeled:
            src_label = read_image(fname=label_fn, imtype=self.LabelType)
            label_exist = make_flag(cls=self.cls_num, labelmap=self.label_map[d_name])
        else:
            label_exist = np.zeros([self.cls_num, 1])

        status = ['tea', 'stu']

        for mode in status:
            t, t_param = generate_transform(rand=self.aug_data)
            image = resample(image=src_image, imtype=self.ImageType, size=self.rs_size, spacing=self.rs_spacing, origin=None, 
                            transform=t, linear=True, dtype=np.float32, use_min_default=True)
            image['array'] = normalize(image['array'], min=self.rs_intensity[0], max=self.rs_intensity[1])
            output['{0:s}_data'.format(mode)] = torch.from_numpy(image['array'])
            output['{0:s}_size'.format(mode)] = image['size']
            output['{0:s}_spacing'.format(mode)] = image['spacing']
            output['{0:s}_origin'.format(mode)] = image['origin']
            output['{0:s}_transform'.format(mode)] = np.array(t_param, dtype=np.float32)

            if labeled:
                label = resample(image=src_label, imtype=self.LabelType, size=self.rs_size, spacing=self.rs_spacing, origin=None, 
                            transform=t, linear=False, dtype=np.int64, use_min_default=True)
                tmp_array = np.zeros_like(label['array'])
                lmap = self.label_map[d_name]
                for key in lmap:
                    tmp_array[label['array'] == key] = lmap[key]
                label['array'] = tmp_array
                label_bin = make_onehot(label['array'], cls=self.cls_num)                
            else:
                label_bin = make_onehot(np.zeros_like(image['array'], dtype=np.int64), cls=self.cls_num)                
            output['{0:s}_label'.format(mode)] = torch.from_numpy(label_bin.astype(np.float32))

        
        output['org_size'] = image['org_size']
        output['org_spacing'] = image['org_spacing']
        output['org_origin'] = image['org_origin']
        output['label_exist'] = label_exist
        output['dataset'] = d_name
        output['case'] = casename
        output['eof'] = True

        return output