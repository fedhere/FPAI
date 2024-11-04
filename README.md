# FPAI

Story board

# Introduction 

  - Proposal motivation
  - Land cover change, Fish Ponds are replacing what on land?
  - Food sustainability: Location of ponds + dietary/nutrition information of nearby regions, increase # ponds $$ dietary needs in local population
  - General detection of areas with fish ponds: model trained with one state was able to detect some ponds on other states

# Fish Pond Census
  - Selection of states
  - Methods used for the census
  - Who made it
    
  ![Figure 1: Map showing Nigeria and states where census took place](images/census.png)

# Data extraction
  - How data was extracted (QGIS, Google satellite XYZ Titles)
  - Characteristics of the data (estimation date of data, size)
    - Total images per state
  - Data not usable for time series analysis

## Data manual Labeling
  - Roboflow
    
![Figure 2: Examples annotated images](images/roboflow_annotations.png)

![Figure 3: Distribution #ponds per image per state](images/dist_ponds_state.png)

![Figure 4: Distribution fish pond sizes](images/dist_ponds_size.png)

# Model Architectures 
## YOLO Model
  - YOLO models
  - YOLOv7
    
  ![Figure 5: YOLOv7 architecture](images/yolov7_arch.png)
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
  - NVDI?
  - Area?
  - Distances?
  - Size?

# Results

![Figure 6: Loss curve per model](images/loss_curve.png)

![Figure 7:IoU vs score, threshold = 0.5 per state](images/iou_vs_score.png)

![Figure 8:Precision-recall curve per state](images/precision-recall.png)

![Figure 9:True area vs Predicted area per state](images/true_predicted_area.png)

# Conclusion






  



