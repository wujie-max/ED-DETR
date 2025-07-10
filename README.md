# An edge-guided dual-branch feature optimization network for enhanced small object detection in UAV images 
Official PyTorch implementation of **ED-DETR**.

Jie Wu, Jinxia Yu, Yongli Tang and Mingfu Zhu


<details>
  <summary>
  <font size="+1">Abstract</font>
  </summary>
Aiming to resolve the problems of low detection accuracy and small object missing in object detection algorithms for UAV images, this paper proposed an edge-guided dual-branch feature optimization network based on RT-DETR (ED-DETR). Firstly, a dual-branch feature extraction and aggregation unit (DFEA) is designed. Different from other traditional methods, two distinct feature extraction branches are used by this unit to separately extract high frequency texture and low frequency structure features, thus solving the feature loss problem caused by their overlap. Meanwhile, reparameterization technology is employed to eliminate the extra computational overhead of dual branches, ensuring the network remains lightweight. Secondly, an edge-guided DFEA module (EDFEA) is proposed. By enhancing the edge features of objects using max pooling technology, an edge-guided unit (EG) is designed, enabling precise edge localization of small objects. Based on the above research, EG is integrated into DFEA to construct the EDFEA module, which further enhances the network's perception of edge features while preserving rich feature representations. Finally, a hybrid loss function, named Mal-Shape loss function, is constructed. By fusing the advantages of Mal loss and ShapeIoU loss, it not only enhances the robustness of low-quality bounding box matching but also incorporates shape-aware and orientation-sensitive mechanisms, enabling precise boundary localization and small object detection in complex scenarios. Experimental results demonstrate that our model outperforms the baseline model by 3.7% and 1.2% in mAP\(_{50}\) on the VisDrone and RSOD datasets, respectively, while maintaining the same computational complexity, thus validating its effectiveness in detecting small objects for UAV images.
</details>

![compare1](https://github.com/user-attachments/assets/5221a063-3f6c-41e6-88e9-b2dc39598d08)

## DataSets

Vistrone2019
(https://github.com/ultralytics/yolov5/releases/download/v1.0/VisDrone2019-DET-train.zip,
          https://github.com/ultralytics/yolov5/releases/download/v1.0/VisDrone2019-DET-val.zip,
          https://github.com/ultralytics/yolov5/releases/download/v1.0/VisDrone2019-DET-test-dev.zip)

It is important to include a script in ultralytics/cfg/datasets/VisDrone. yaml that converts Vistrone2019 to YOLO format.

## Installation
`conda` virtual environment is recommended. 
```
conda create -n EDDETR python=3.10
conda activate EDDETR
pip install -r requirements.txt
pip install -e .
```
## Performance

Vistrone2019

| Model          | Test Size | Params | mAP_50(%) |  mAP_50:95(%)   | 
|:---------------| :-------: |:-------:|:--------:|:--------------:|
| YOLOX-M    |   640     |   25.29 | 	34.5   |   	20.9  |
| YOLOv8m    |    640    |	25.86  |	42.3   |	25.8    |
| YOLOv9m      |   640 | 	20.17 	|44.5	  |26.62    |
| YOLOv10m |    640    |  20.14  |	42.1	 | 27.27      | 
| YOLOv11m      |    640    |  20.15 |	44.4  |	27.2     | 
| YOLOv12m     |    640    | 21.61 	|  43.5 	|  26.5    | 
| PPYOLOE-l    |    640    | 52.26   |	47.9	   |31.6       |
| RT-DETR-r18    |    640    | 20.0 |	46.2	|  29.03      |
| UAV-DETR    |    640    |  20.3	|48.6	  |29.8        |
| ED-DETR(Our)  |    640    | 20.1 |  	**49.9** 	|**32.0**       |

We also tested the performance of other models on the PR curve as shown in the figure below:
![pr_curves_visdrone](https://github.com/user-attachments/assets/3fbc151c-a0b6-4de9-a145-8bf08e6ac471)


## Training 
```
python train.py
```

## Validation
Note that a smaller confidence threshold can be set to detect smaller objects or objects in the distance. 
```
python val.py
```


## Acknowledgement

The code base is built with [ultralytics](https://github.com/ultralytics/ultralytics).

Thanks for the great implementations! 

## Citation
If our code or models help your work, please cite our paper:

```
Our article has not been published yet, we will update it once it is published.
```
