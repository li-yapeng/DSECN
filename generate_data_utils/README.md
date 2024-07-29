# The code for generating the CUB dataset is provided as a reference to understand the dataset generation process.

## Download Original Data
* download images of CUB dataset from the [(link)](https://www.vision.caltech.edu/datasets/cub_200_2011/). Folders are organized according to the following directory structure.
```
image_folder    # the save path of the image
└── CUB_200_2011/
    ├── images
        ├── 001.Black_footed_Albatross
        ...
        └── 200.Common_Yellowthroat
    ...
    └──classes.txt
    
``` 
* download "proposed split version 2.0" from the [(link)](https://www.mpi-inf.mpg.de/departments/computer-vision-and-machine-learning/research/zero-shot-learning/zero-shot-learning-the-good-the-bad-and-the-ugly).
```
feature_folder    # the save path of the feature
└── xlsa17/
    └── data
        ├── CUB
            ├──att_splits.mat
            ...
            └──res101.mat
        ...
        └── SUN
            ├──att_splits.mat
            ...
            └──res101.mat
``` 
## Extract Resnet101 Visual Feature
```
python Extract_ResNet101_Feature.py --image_folder image_folder --feature_folder feature_folder
``` 
**image_folder** and **feature_folder** denote the save path of images and split, respectively.

## Extract Clip Semantic Feature
```
python Extract_CLIP_Semantic.py --feature_folder feature_folder
```

## More Info
If you want to generate semantic representation features of other language models, please refer to the process of extracting clip representation. 

## Note
In order to facilitate mutual comparison, it is recommended that you directly use the public data set extracted in this article.