# FPAI

Story board

# Introduction 

  - Proposal motivation
  - Land cover change, Fish Ponds are replacing what on land?
  - Food sustainability: Location of ponds + dietary/nutrition information of nearby regions, increase # ponds $\propto$ dietary needs in local population
  - General detection of areas with fish ponds: model trained with one state was able to detect some ponds on other states

# Fish Pond Census
  - Selection of states
  - Methods used for the census
  - Who made it
    
  ![Figure 1: Map showing Nigeria and states where census took place](images/map_big.png)
  Figure 1: Nigeria and its 36 geopolitical states. In color are the states considered for this work and where we have information of Fish Ponds location from the WorldFish census.

  ![Figure 2: Fish Pond Location in Ogun State ](images/ogun.png)
  
  Figure 2: In bright green the boundary of Ogun state and in light blue the Fish Pond location. The location are later used to extract RGB images containing Fish Ponds.

# Data extraction
  - How data was extracted (QGIS, Google satellite XYZ Titles)
  - Characteristics of the data (estimation date of data, size)
    - Total images per state
  - Time analysis limitations of google satellite XYZ titles data

 ![Figure 3: Methodology data extraction](images/data_extraction.png)

  Figure 3: Graph showing WorldFish Census -> Fish Ponds location per state -> Construction of Buffer around location -> Extraction of RGB images of size 2,000 x 2,000 pixels using: Python + QGIS + XYZ Tiles Google Satellite

## Data manual Labeling
  - Roboflow
    
![Figure 4: Examples annotated images](images/roboflow_annotations.png)
Figure 2: Examples annotated images in Ogun State

![Figure 5: Distribution #ponds per image per state](images/dist_ponds_state.png)
Figure 3: Distribution number of Fish Ponds per image in all the states considered. There are on average XX amount of annotated Fish Ponds per image; the image with less Fish Ponds has XX and the one with more has XX annotated Fish Ponds. For Ogun states the team annotated 4,951 across X number of images. Delta state 2,981 across Y number of images, other states (maybe a table if we have several states.)

![Figure 6: Distribution fish pond sizes](images/dist_ponds_size.png)
Figure 4: Distribution of annotated Fish Pond area in m2 across all the states considered in this work. In average the area of the Fish Pond is XX m2, with some outliers whose size is >5,000m2.


# Model Architectures 
## YOLO Model
  - YOLO models
  - YOLOv7
    

   ![Figure 7: YOLOv7 architecture](images/yolov7_arch.jpg)
  Figure 7: YOLOv7 architecture, figure extracted from https://doi.org/10.3389/fpls.2023.1211075 

  
## SAM?

# Methodology
## Description of models
  - Model per state
  - Fine tuning strategies

# Training Details and Metrics
  - Darwin UDEL, GPU
  - Split train/test/valid
  - IoU, Average Precision
  - TP, FP, FN

    
 ![Figure 8: Methodology ](images/metodoly.png)
  
  Figure 8: Graph showing Methodology -> Extracted RGB images per states -> Roboflow Annotation -> Train/Valid/Test split -> Fine Tune per state the state-of-the-art instance segmentation model, the YOLOv7 -> output: masks -> masks to pixel polygon -> pixel to coordinates



## Ogun state
  - Total images, Split train/test/valid, number epochs
## Fine-tuning Delta state
  - Total images, Split train/test/valid, number epochs, fine tune strategie
## Fine-tuning Kwara state
  - Total images, Split train/test/valid, number epochs, fine tune strategie
## Trained all
  - Total images, Split train/test/valid, number epochs, fine tune strategie
## By geopolitica regions?

# Post-processing tasks for removing False Positives
  - NDVI?
  - Area?
  - Distances?
  - Size?

# Results

![Figure 8: Loss curve per model](images/loss_curve.png)
Figure 8: Test loss curve for each states considered here. 

![Figure 9:Precision-recall curve per state](images/precision-recall.png)
Figure 9: Precision-Recall curve at different IoU thresholds for SS state. 

![Figure 9a:True area vs Predicted area per state](images/true_predicted_area.png)
Figure 9a: True area vs Predicted area for the test data set for all the states considered in the work. The model is able to predict the Fish Ponds area distribution, including some of the outliers with larger sizes. 

![Figure 9b: Prediction and True labels colored coded by classification class](images/area_dist_ogun_pred.png)
Figure 9b: Prediction and True labels colored coded by classification class. False Positive predictions are embeded within the true area distribution, the model is predicting the correct size but not detecting true Fish Ponds.


# Conclusion






  



