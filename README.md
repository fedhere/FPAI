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
    
![Figure 2: Examples annotated images](images/roboflow_annotations.png)
Figure 2: Examples annotated images in Ogun State

![Figure 3: Distribution #ponds per image per state](images/dist_ponds_state.png)
Figure 3: Distribution number of Fish Ponds per image in all the states considered. There are on average XX amount of annotated Fish Ponds per image; the image with less Fish Ponds has XX and the one with more has XX annotated Fish Ponds. For Ogun states the team annotated 4,951 across X number of images. Delta state 2,981 across Y number of images, other states (maybe a table if we have several states.)

![Figure 4: Distribution fish pond sizes](images/dist_ponds_size.png)
Figure 4: Distribution of annotated Fish Pond area in m2 across all the states considered in this work. In average the area of the Fish Pond is XX m2, with some outliers whose size is >5,000m2.


# Model Architectures 
## YOLO Model
  - YOLO models
  - YOLOv7
    
  ![Figure 5: YOLOv7 architecture](images/yolov7_arch.jpg)
  Figure 5: YOLOv7 architecture, figure extracted from https://doi.org/10.3389/fpls.2023.1211075 
  
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

![Figure 6: Loss curve per model](images/loss_curve.png)
Figure 6: Loss curve per model

![Figure 7:IoU vs score, threshold = 0.5 per state](images/iou_vs_score.png)

![Figure 8:Precision-recall curve per state](images/precision-recall.png)

![Figure 9:True area vs Predicted area per state](images/true_predicted_area.png)
Figure 9:True area vs Predicted area per state


# Conclusion






  



