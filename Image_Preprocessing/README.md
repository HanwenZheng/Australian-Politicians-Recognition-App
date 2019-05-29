# Image Preprocessing - Australian Politicians Recognition
## Working Environment
Codes in this folder assumes you are using the following environment:

- Python 3.7.3
- Anaconda 3
## Dependencies
Codes in this folder depends on the following libraries:

- numpy 1.16.3
- opencv  3.4.2
- scikit-image 0.15.0
> Use `conda install` to install missing libraries
## Augmentation Operations
- [cv2dnn_crop.py](https://github.com/HanwenZheng/PoliticiansAU_Recognition/blob/master/Image_Preprocessing/cv2dnn_crop.py "cv2dnn_crop.py"): Used to batch extract faces from images.
- [random_crop.py](https://github.com/HanwenZheng/PoliticiansAU_Recognition/blob/master/Image_Preprocessing/random_crop.py "random_crop.py"): Used to generate random cropped images.
- [gamma_adjust.py](https://github.com/HanwenZheng/PoliticiansAU_Recognition/blob/master/Image_Preprocessing/gamma_adjust.py "gamma_adjust.py"): Used to generate gamma adjusted images.
- [image_mirror.py](https://github.com/HanwenZheng/PoliticiansAU_Recognition/blob/master/Image_Preprocessing/image_mirror.py "image_mirror.py"): Used to generate mirrored images.
