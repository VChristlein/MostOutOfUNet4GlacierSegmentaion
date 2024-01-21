# How to Get the Most Out of U-Net for Glacier Calving Front Segmentation

Code by Maniraman Periyasamy. All credits to him. 

## Abstract
The melting of ice sheets and glaciers is one of the main contributors to global sea-level rise. Hence, continuous monitoring of glacier changes and in particular the mapping of positional changes of their calving front is of significant importance. This delineation process, in general, has been carried out manually, which is time-consuming and not feasible for the abundance of available data within the past decade. Automatic delineation of the glacier fronts in synthetic aperture radar (SAR) images can be performed using deep learning-based U-Net models. This article aims to study and survey the components of a U-Net model and optimize the model to get the most out of U-Net for glacier (calving front) segmentation. We trained the U-Net to segment the SAR images of Sjogren-Inlet and Dinsmoore–Bombardier–Edgworth glacier systems on the Antarctica Peninsula region taken by ERS-1/2, Envisat, RadarSAT-1, ALOS, TerraSAR-X, and TanDEM-X missions. The U-Net model was optimized in six aspects. The first two aspects, namely data preprocessing and data augmentation, enhanced the representation of information in the image. The remaining four aspects optimized the feature extraction of U-Net by finding the best-suited loss function, bottleneck, normalization technique, and dropouts for the glacier segmentation task. The optimized U-Net model achieves a dice coefficient score of 0.9378 with a 20% improvement over the baseline U-Net model, which achieved a score of 0.7377. This segmentation result is further postprocessed to delineate the calving front. The optimized U-Net model shows 23% improvement in the glacier front delineation compared to the baseline model.

## Cite
```
@ARTICLE{9699423,
  author={Periyasamy, Maniraman and Davari, Amirabbas and Seehaus, Thorsten and Braun, Matthias and Maier, Andreas and Christlein, Vincent},
  journal={IEEE Journal of Selected Topics in Applied Earth Observations and Remote Sensing}, 
  title={How to Get the Most Out of U-Net for Glacier Calving Front Segmentation}, 
  year={2022},
  volume={15},
  number={},
  pages={1712-1723},
  doi={10.1109/JSTARS.2022.3148033},
  ISSN={2151-1535}
}
```

**Code description**: Please see Readme.pdf
