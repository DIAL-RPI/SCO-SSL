import os, sys
import numpy as np
import pydicom
import SimpleITK as sitk
import vtk
from vtkmodules.util import vtkImageExportToArray

# read in an MRI image from DICOM files
# input: 
#     'dcm_dir': directory of DICOM series
# output: 
#     'image': SimpleITK image of MRI image
#     'series_uid': Series UID of DICOM series
def read_mr_from_dcm_dir(dcm_dir):
    series_uid = None
    fn_map = {}
    for fn in os.listdir(dcm_dir):
        if not fn.endswith('.dcm'):
            continue
        dataset = pydicom.dcmread('{}/{}'.format(dcm_dir, fn), force=True)
        if series_uid is None:
            series_uid = dataset.SeriesInstanceUID
            width = dataset.Columns
            height = dataset.Rows
        if series_uid != dataset.SeriesInstanceUID:
            continue
        fn_map[int(dataset.InstanceNumber)] = fn

    length = len(fn_map)
    image_array = np.zeros([length, height, width], dtype=np.int16)
    slice_id = 0
    for i in sorted(fn_map):
        fn = fn_map[i]
        dataset = pydicom.dcmread('{}/{}'.format(dcm_dir, fn), force=True)
        if slice_id == 0:
            origin = np.array(dataset.ImagePositionPatient)
            spacing = np.zeros_like(origin)
            spacing[0] = dataset.PixelSpacing[1]
            spacing[1] = dataset.PixelSpacing[0]
            spacing[2] = dataset.SliceThickness

        image_array[slice_id,:,:] = dataset.pixel_array
        slice_id += 1
    
    image = sitk.GetImageFromArray(image_array)
    image.SetSpacing(spacing)
    image.SetOrigin(origin)

    return image, series_uid

# read in an ultrasound image volume from DICOM files
# input: 
#     'dcm_dir': directory of DICOM series
# output: 
#     'image': SimpleITK image of ultrasound image
#     'series_uid': Series UID of DICOM series
#     'is_us': a boolean flag (== True if image modality is 'US')
def read_us_from_dcm_dir(dcm_dir):
    for fn in os.listdir(dcm_dir):
        if not fn.endswith('.dcm'):
            continue
        dataset = pydicom.dcmread('{}/{}'.format(dcm_dir, fn), force=True)
        series_uid = dataset.SeriesInstanceUID
        length = dataset.NumberOfFrames
        height = dataset.Rows
        width = dataset.Columns
        is_us = (dataset.Modality == 'US')
        image_array = np.zeros([length, height, width], dtype=np.int8)
        image_array = dataset.pixel_array
        image_array = np.transpose(image_array, (2,0,1))
        image_array = np.flip(image_array, axis=1)
        image_array = np.flip(image_array, axis=0)

        spacing = np.zeros(3, dtype=np.float64)
        if ('PixelSpacing' in dataset) and ('SliceThickness' in dataset):
            spacing[0] = dataset.PixelSpacing[0]
            spacing[1] = dataset.SliceThickness
            spacing[2] = dataset.PixelSpacing[1]
        else:
            spacing[0] = dataset[0x11291016].value
            spacing[1] = dataset[0x11291016].value
            spacing[2] = dataset[0x11291016].value
        origin = np.zeros(3, dtype=np.float64)
        origin[0] = -spacing[0] * image_array.shape[2] * 0.5
        origin[1] = -spacing[1] * image_array.shape[1] * 0.5
        origin[2] = -spacing[2] * image_array.shape[0] * 0.5

        image = sitk.GetImageFromArray(image_array)
        image.SetSpacing(spacing)
        image.SetOrigin(origin)
        break

    return image, series_uid, is_us

def write_image_to_file(image, fn):
    writer = sitk.ImageFileWriter()
    writer.SetFileName(fn)
    writer.Execute(image)

def loadStl(fn):
    """Load the given STL file, and return a vtkPolyData object for it."""
    reader = vtk.vtkSTLReader()
    reader.SetFileName(fn)
    reader.Update()
    polydata = reader.GetOutput()
    return polydata

# read in a segmentation label volume from STL file
# input: 
#     'fn': STL filename
#     'ref_image': a reference image providing meta information such as image size, spacing, and origin
# output: 
#     'label': SimpleITK image of label mask
def read_volume_from_stl(fn, ref_image):
    polydata = loadStl(fn)

    size = ref_image.GetSize()
    spacing = ref_image.GetSpacing()
    origin = ref_image.GetOrigin()

    whiteImage = vtk.vtkImageData()
    whiteImage.SetSpacing(spacing[0], spacing[1], spacing[2])
    whiteImage.SetDimensions(size[0], size[1], size[2])
    whiteImage.SetExtent(0, size[0] - 1, 0, size[1] - 1, 0, size[2] - 1)
    whiteImage.SetOrigin(origin[0], origin[1], origin[2])
    whiteImage.AllocateScalars(vtk.VTK_UNSIGNED_CHAR, 1)
    whiteImage.GetPointData().GetScalars().Fill(1)

    pol2stenc = vtk.vtkPolyDataToImageStencil()
    pol2stenc.SetInputData(polydata)
    pol2stenc.SetOutputOrigin(origin[0], origin[1], origin[2])
    pol2stenc.SetOutputSpacing(spacing[0], spacing[1], spacing[2])
    pol2stenc.SetOutputWholeExtent(whiteImage.GetExtent())
    pol2stenc.Update()

    imgstenc = vtk.vtkImageStencil()
    imgstenc.SetInputData(whiteImage)
    imgstenc.SetStencilConnection(pol2stenc.GetOutputPort())
    imgstenc.ReverseStencilOff()
    imgstenc.SetBackgroundValue(0)
    imgstenc.Update()

    v2a = vtkImageExportToArray.vtkImageExportToArray()
    v2a.SetInputData(imgstenc.GetOutput())
    array = v2a.GetArray()

    label = sitk.GetImageFromArray(array)
    label.SetSpacing(spacing)
    label.SetOrigin(origin)

    return label




if __name__ == '__main__':

    # This Python script implements the data conversion (from DICOM/STL to NIfTI format) specially designed for UCLA prostate biopsy dataset
    # You can get the UCLA prostate biopsy dataset from: https://doi.org/10.7937/TCIA.2020.A61IOC1A
    
    # Note: it is recommended to keep the original file/folder names unchanged so that you can directly use this script to do the data conversion
    #       otherwise, you will need to change the file/folder names referred in the code

    src_dcm_root = '{}/source/Prostate-MRI-US-Biopsy'.format(sys.path[0]) # source directory of DICOM image files (we kept the folder name unchanged from the original UCLA dataset)
    src_stl_root = '{}/source/STLs/STLs'.format(sys.path[0]) # source directory of STL label files (we kept the folder name unchanged from the original UCLA dataset)
    dst_image_root = '{}/image'.format(sys.path[0]) # output directory to store the converted images (in NIfTI format)
    dst_label_root = '{}/label'.format(sys.path[0]) # output directory to store the converted labels (in NIfTI format)

    case_num = 0
    mr_im_num = 0
    mr_lb_num = 0
    us_im_num = 0
    us_lb_num = 0
    for dirname1 in os.listdir(src_dcm_root):
        if 'Prostate-MRI-US-Biopsy-' in dirname1:
            case_id = int(dirname1.split('Prostate-MRI-US-Biopsy-')[1])
        else:
            continue
        casename = 'Case{0:04d}'.format(case_id)
        case_dir = '{}/{}'.format(src_dcm_root, dirname1)
        print(case_id, casename)
        case_num += 1

        image_dir = '{}/{}'.format(dst_image_root, casename)
        os.makedirs(image_dir, exist_ok=True)
        label_dir = '{}/{}'.format(dst_label_root, casename)
        os.makedirs(label_dir, exist_ok=True)

        mr_num = 0
        us_num = 0
        for dirname2 in os.listdir(case_dir):
            if 'MR' in dirname2:
                mr_num += 1
                line = 'MR {}'.format(mr_num)
                for dirname3 in os.listdir('{}/{}'.format(case_dir, dirname2)):
                    image, series_uid = read_mr_from_dcm_dir('{}/{}/{}'.format(case_dir, dirname2, dirname3))
                    write_image_to_file(image, '{0:s}/mr_{1:d}.nii.gz'.format(image_dir, mr_num))
                    mr_im_num += 1
                    line += ' mr_image'
                    break

                stl_fn = '{0:s}/Prostate-MRI-US-Biopsy-{1:04d}-ProstateSurface-seriesUID-{2:s}.STL'.format(src_stl_root, case_id, series_uid)
                if os.path.isfile(stl_fn):
                    label = read_volume_from_stl(stl_fn, image)
                    write_image_to_file(label, '{0:s}/mr_{1:d}.nii.gz'.format(label_dir, mr_num))
                    mr_lb_num += 1
                    line += ' mr_label'
                else:
                    line += ' no_label'

            else:
                us_num += 1
                line = 'US {}'.format(us_num)
                for dirname3 in os.listdir('{}/{}'.format(case_dir, dirname2)):
                    image, series_uid, is_us = read_us_from_dcm_dir('{}/{}/{}'.format(case_dir, dirname2, dirname3))
                    write_image_to_file(image, '{0:s}/us_{1:d}.nii.gz'.format(image_dir, us_num))
                    us_im_num += 1
                    if is_us:
                        line += ' us_image'
                    else:
                        line += ' non_us_image'
                    break

                stl_fn = '{0:s}/Prostate-MRI-US-Biopsy-{1:04d}-ProstateSurface-seriesUID-{2:s}.STL'.format(src_stl_root, case_id, series_uid)
                if os.path.isfile(stl_fn):
                    label = read_volume_from_stl(stl_fn, image)
                    write_image_to_file(label, '{0:s}/us_{1:d}.nii.gz'.format(label_dir, us_num))
                    us_lb_num += 1
                    line += ' us_label'
                else:
                    line += ' no_label'
            print(line)
                
    print('Case num:', case_num)
    print('MR image num:', mr_im_num)
    print('MR label num:', mr_lb_num)
    print('US image num:', us_im_num)
    print('US label num:', us_lb_num)