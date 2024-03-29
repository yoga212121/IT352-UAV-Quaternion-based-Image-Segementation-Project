# IT352 Course Project : Image Segmentation using quaternions of Clifford Algebra

GitHub repository link: https://github.com/yoga212121/IT352-UAV-Quaternion-based-Image-Segementation-Project
Colab Notebook link: https://colab.research.google.com/drive/1sESRCjKNiTzHKzDJjxIoo8EF9lmZi2gH?usp=sharing

## Journal Paper:

Our project is the implementation of the paper:
UAVs_Agricultural_Image_Segmentation_Predicated_by_Clifford_Geometric_Algebra
Citation : Khan, Prince Waqas, et al. "UAV’s agricultural image segmentation predicated by clifford geometric algebra." Ieee Access 7 (2019): 38442-38450.

## Dependencies:

- Flask (v2.0.2 onwards)
- OpenCV (cv2 4.5.3 or later) for image processing
- NumPy (v1.21.0 or later) for numerical computation
- Scikit-learn (v0.24.2 or later) for principal component analysis (PCA)

## Directory Structure

The directory contains the following files and folders

- app.py: It contains a Flask web application that processes uploaded images by applying quaternion-based masking, principal component analysis (PCA)-based segmentation, and normalization. The processed images are saved and displayed back to the user through the application's web interface.
- get-pip.py: Python script designed to bootstrap and install the latest version of pip and setuptools from PyPI, ensuring proper dependency management and package installation. It includes functionalities to determine the installation arguments, patch the installation process to provide default certificates, and execute the pip installation within a temporary working directory.
- webpages: contains 2 html files index.html: specifying the UI for input image retrieval and processing, and result.html: displays the final image after segmentation
- uploads: contains experimental raw images to be segmented
- results: contains segmented images for the corresponding experimental images
