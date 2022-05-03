# TPE-Det

This project is developed to detect cerebral microbleeds (CMBs) from brain MR images. 

To do so, we have proposed a end-to-end TPE-Det (Triplanar Ensemble Detection Network) with only a single-stage utilizing the ensemble of 2D CNN-based networks. 

First, we independently detect the CMBs using axial, sagittal, and coronal images via EfficientDet, one of the state-of-the-art 2D object detection networks. We exploit MR susceptibility-weighted imaging (SWI) and phase images as two-channel inputs for the network. 

Then, by calculating and comparing the three-dimensional coordinates of each candidate from each plane, the detected candidates on all three planes are considered to be our final detections.

# CMBs-Candidate-Detection-Via-EfficientDet:

We employed EfficientDet for our 2D detection networks.

The github repos for EfficientDet:

EfficientDet-PyToch: https://github.com/rwightman/efficientdet-pytorch

PyTorch-Image-Models: https://github.com/rwightman/pytorch-image-models

Here, we make a complete python code for training EfficientDet and performing inference in "TPE-Det.py".
You can easily change the hyper-parameters such as learning rate, number of epochs, batch size, and selecting appropriate loss function and optimizer.

# Ensembling the detection networks of three different perpendicular planes:

Here, we make a complete python code for the second stage available for researchers.

The file named "CMB_3DCNN.py" contains all processing steps: data reading, 3D network generation, training and testing the model, and saving the network weights and predicted labels.
You can easily change the hyper-parameters such as learning rate, number of epochs, batch size, and selecting appropriate loss function and optimizer.

# Dataset:
The dataset used in this project has been collected with a collaboration between the Medical Imaging LABoratory (MILAB) at Yonsei University and Gachon University Gil Medical Center.

Our dataset contains two in-plane resolutions as follows:
1. High in-plane resolution (HR): 0.50x0.50 mm^2, and
2. Low in-plane resolution (LR): 0.80x0.80 mm^2.

HR data composites of 72 subjects, while LR data contains 107 subjects.
All Data contain SWI, Phase and Magnitude images.

The Label folder involves excel files, where each excel file is with the same name as data in the Data folder.
The information of the location of cerebral microbleeds (CMBs) in brain images exist in those excel files as follow:
- 1st column represents the slice number of subject,
- 2nd and 3rd columns indicate the x(column-wise) and y(row-wise) pixel location of CMB in that slice, respectively.

** Source code is provided. However, due to patent transfer issue, no longer able to provide the data.

# Published-Paper:

[1] Mohammed A. Al-masni, Woo-Ram Kim, Eung Yeop Kim, Young Noh, and Dong-Hyun Kim “Automated Detection of Cerebral Microbleeds in MR Images: A Two-Stage Deep Learning Approach,” NeuroImage: Clinical, vol. 28, pp. 102464, 2020. [ELSEVIER Publisher]

[2] Mohammed A. Al-masni, Woo-Ram Kim, Eung Yeop Kim, Young Noh, and Dong-Hyun Kim, “A Two Cascaded Network Integrating Regional–based YOLO and 3D-CNN for Cerebral Microbleeds Detection,” in the 42nd Annual International Conferences of the IEEE Engineering in Medicine and Biology Society (EMBC), pp. 1055-1058. IEEE, Montreal, Quebec, Canada, 2020.

When using our code or dataset for research publications, please cite our papers.
