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
The file named "TPE-Det.py" contains processing steps for ensembling detection networks by final bounding box selection.

Initially, we independently detected the CMBs using the axial, sagittal, and coronal images via the proposed TPE-Det. 

Then, the three-dimensional coordinates of each detected candidate were calculated from its x and y coordinates and the slice number of the image. 

Afterward, we calculated the Euclidean distances between the candidates detected from each plane and deemed the objects to be single CMB if they were closer than a pre-set distance threshold value. 

Based on this method, the candidates detected on all three planes were considered to be our final detections. 

# Dataset:

Two datasets (DS1 & DS2) were used in this project. DS1 was collected from Gachon University Gil Medical Center, the Republic of Korea. The SWI data were acquired using 3.0 T Verio and Skyra Siemens MRI scanners (Siemens Healthineers, Germany) with the following imaging parameters: a resolution of 0.5×0.5×2.0 mm3, a repetition time (TR) of 27 ms, an echo time (TE) of 20 ms, a flip angle (FA) of 15°, and a field of view (FOV) of 256×224×144 mm3. A total of 116 subjects with 367 microbleeds and 12 subjects not including CMBs were collected.

Additionally, to evaluate our approach’s generalizability, we collected another dataset (DS2) from Seoul National University Hospital (SNUH), the Republic of Korea, with different acquisition parameters. The SWI data were acquired using a 3.0 T Biograph mMR Siemens MRI scanner (Siemens Healthineers, Germany) with the following protocol: a resolution of 0.5×0.5×3.0 mm3, a TR of 28 ms, a TE of 20 ms, an FA of 15°, and an FOV of 192×219×156 mm3. Note that this slice thickness (i.e.,3.0 mm) is commonly used in routine clinical practice (MICCAI, 2021). A total of 58 subjects with 148 microbleeds and 21 subjects absent from CMBs were collected. 

Both studies were approved by the individual Institutional Review Board (IRB) of Gachon University Gil Medical Center and SNUH. 

** The utilized MRI data cannot be made openly available due to the privacy issues of clinical data.

<!-- # Published-Paper:

[1] Mohammed A. Al-masni, Woo-Ram Kim, Eung Yeop Kim, Young Noh, and Dong-Hyun Kim “Automated Detection of Cerebral Microbleeds in MR Images: A Two-Stage Deep Learning Approach,” NeuroImage: Clinical, vol. 28, pp. 102464, 2020. [ELSEVIER Publisher]

[2] Mohammed A. Al-masni, Woo-Ram Kim, Eung Yeop Kim, Young Noh, and Dong-Hyun Kim, “A Two Cascaded Network Integrating Regional–based YOLO and 3D-CNN for Cerebral Microbleeds Detection,” in the 42nd Annual International Conferences of the IEEE Engineering in Medicine and Biology Society (EMBC), pp. 1055-1058. IEEE, Montreal, Quebec, Canada, 2020.

When using our code or dataset for research publications, please cite our papers. -->
