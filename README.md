# This code is for deformable-detr using support image

## The file tree should be:
```
-work/

​	-deformable_detr_support/

-data/

​	-cityscape/
```

## Preparation

### dataset and pretrained weights

The custome cityscape dataset is in the mega cloud, user should download from 

https://mega.nz/file/wXcTWQoR#ANd1kLJe1hh2A0LwHwUHZAfpAGVmyYWr4GHb0UiMd6I

There are two checkpoints for visualization, one for heat map and the other for detection results, download link:

For bounding box generation:

https://mega.nz/file/hDlhjCLR#OIlmslbyLkLCqmuYC4ZXfdpZSdByBfQX-CplUrA2sdA

For heatmap generation:

https://mega.nz/file/Mf0B1YoR#1Y36nWxgJTl5mZq-npiOOg9qONgDf6XCOclosgnd37o

After downloading the dataset, please create two soft links ./car_pool -> data/cityscape/car and ./person_pool -> data/cityscape/person in -work/deformable_detr_support.

### model
After placing the model file in the correct location, please install the MultiScaleDeformableAttention by executing:

```
cd work/deformable_detr_support
./models/opt/make.sh
```

You should modify the checkpoint path in each .sh file before executing it.

## Running
### Visualizing detection results:
```
./generate_pred_bbox.sh
```


## visualizing heatmap with reference points:
```
./generate_car_heat_map.sh
```

You could find the results in the output dictionary which is specified in each .sh file. 




