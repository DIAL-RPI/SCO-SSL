# Shadow-consistent Semi-supervised Learning (SCO-SSL)
Shadow-consistent Semi-supervised Learning for Prostate Ultrasound Segmentation

This is a python (PyTorch) implementation of **Shadow-consistent Semi-supervised Learning (SCO-SSL)** method for prostate ultrasound segmentation proposed in our ***IEEE Transactions on Medical Imaging*** journal paper [**"Shadow-consistent Semi-supervised Learning for Prostate Ultrasound Segmentation"**](https://doi.org/10.1109/TMI.2021.3139999).

## Citation
  *X. Xu, T. Sanford, B. Turkbey, S. Xu, B. J. Wood and P. Yan, "Shadow-Consistent Semi-Supervised Learning for Prostate Ultrasound Segmentation," in IEEE Transactions on Medical Imaging, vol. 41, no. 6, pp. 1331-1345, June 2022, doi: 10.1109/TMI.2021.3139999.*

    @article{Xu2021SCOSSL,
      title={Shadow-consistent Semi-supervised Learning for Prostate Ultrasound Segmentation}, 
      author={Xu, Xuanang and Sanford, Thomas and Turkbey, Baris and Xu, Sheng and Wood, Bradford J. and Yan, Pingkun},
      journal={IEEE Transactions on Medical Imaging}, 
      year={2022},
      volume={41},
      number={6},
      pages={1331-1345},
      publisher={IEEE},
      doi={10.1109/TMI.2021.3139999}
    }

## Update
  - **Sep 15, 2022**: Add a script [`UCLA_data_conversion.py`](https://github.com/DIAL-RPI/SCO-SSL/blob/main/UCLA_data_conversion.py) for **data format conversion** (from *DICOM/STL* to *NIfTI* format) specially designed for [**UCLA Prostate-MRI-US-Biopsy dataset**](https://doi.org/10.7937/TCIA.2020.A61IOC1A) that was used in our paper.
  - **Mar 23, 2022**: Add a script [`3d_dist_visual.py`](https://github.com/DIAL-RPI/SCO-SSL/blob/main/3d_dist_visual.py) for **3D distance error visualization** that was shown in our paper.

## Abstract
Prostate segmentation in transrectal ultrasound (TRUS) image is an essential prerequisite for many prostate-related clinical procedures, which, however, is also a long-standing problem due to the challenges caused by the low image quality and shadow artifacts. In this paper, we propose a Shadow-consistent Semi-supervised Learning (SCO-SSL) method with two novel mechanisms, namely shadow augmentation (Shadow-AUG) and shadow dropout (Shadow-DROP), to tackle this challenging problem. Specifically, Shadow-AUG enriches training samples by adding simulated shadow artifacts to the images to make the network robust to the shadow patterns. Shadow-DROP enforces the segmentation network to infer the prostate boundary using the neighboring shadow-free pixels. Extensive experiments are conducted on two large clinical datasets (a public dataset containing 1,761 TRUS volumes and an in-house dataset containing 662 TRUS volumes). In the fully-supervised setting, a vanilla U-Net equipped with our Shadow-AUG&Shadow-DROP outperforms the state-of-the-arts with statistical significance. In the semi-supervised setting, even with only 20% labeled training data, our SCO-SSL method still achieves highly competitive performance, suggesting great clinical value in relieving the labor of data annotation. Source code is released at [https://github.com/DIAL-RPI/SCO-SSL](https://github.com/DIAL-RPI/SCO-SSL).

## Method
### Scheme of SCO-SSL
<img src="./fig1.png"/>

### Scheme of Shadow augmentation (Shadow-AUG)
<img width="600" src="./fig2.png"/>

### Scheme of Shadow dropout (Shadow-DROP)
<img width="600" src="./fig3.png"/>

## Data
You can run our code using the public dataset, [**UCLA Prostate-MRI-US-Biopsy dataset**](https://doi.org/10.7937/TCIA.2020.A61IOC1A), shared by the Institute of Urologic Oncology, University of California-Los Angeles (UCLA) on [**the Cancer Imaging Archive (TCIA)**](https://www.cancerimagingarchive.net/) platform. We provided a Python script [`UCLA_data_conversion.py`](https://github.com/DIAL-RPI/SCO-SSL/blob/main/UCLA_data_conversion.py) to convert the original data to the NIfTI format that was accepted by our dataloader interface.

## Contact
You are welcome to contact us:  
  - [xux12@rpi.edu](mailto:xux12@rpi.edu) ([Dr. Xuanang Xu](https://superxuang.github.io/))
