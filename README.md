# Copy-Pasting Coherent Depth Regions Improves Contrastive Learning for Urban-Scene Segmentation
## Overall Pipeline 
![Pipeline](https://github.com/LeungTsang/CPCDR/raw/main/fig/pipeline.png)
## Visualization  
![Visualization](https://github.com/LeungTsang/CPCDR/raw/main/fig/fig1.gif)

## The expected directory of dataset(Cityscapes)  
~~~  
dataset for training  
      └── rgb/  
      │     └── leftImg8bit/  
      │           └── train/  
      │                   ├── aachen/  
      │                   │     ├── aachen_000000_000000_leftImg8bit.png  
      │                   │     └── ...  
      │                   └── ...  
      ├── disp/  
      │     └── leftImg8bit/  
      │           └── train/  
      │                   ├── aachen/  
      │                   │     ├── aachen_000000_000000_leftImg8bit.png  
      │                   │     └── ...  
      │                   └── ...  
      └── region/  
            └── leftImg8bit/  
                  └── train/  
                          ├── aachen/  
                          │     ├── aachen_000000_000000_leftImg8bit.png  
                          │     └── ...  
                          └── ...  
prepared dataset for training 
      └── config.pth     
      └── epoch0/  
      │     ├── 0000000000_img.png
      │     ├── 0000000000_label.png
      │     ├── 0000000000_transforms.pt
      │     └── ...
      └── ... 
      
dataset for evaluation  
      └── leftImg8bit/  
      │           └── val/  
      │                   ├── frankfurt/  
      │                   │     ├── frankfurt_000000_000294_leftImg8bit.png  
      │                   │     └── ...  
      │                   └── ...  
      └── gtFine/  
                  └── val/  
                          ├── frankfurt/  
                          │     ├── frankfurt_000000_000294_gtFine_labelIds.png  
                          │     └── ...  
                          └── ...  
~~~  

## Simple Usage  
### Training  

~~~  
python run.py --data_path path_to_dataset_for_training  
~~~  
generating training data by copy-paste in real time can be time-consuming, you might generate prepared data for repeated use
~~~
python run.py --data_path path_to_dataset_for_training  --only_generate_data
~~~
use preparead data for training
~~~
python run.py --data_path prepared dataset for training  --use_prepared_data
~~~
### Evaluation  
~~~  
python eval_unsupervised.py --model_path path_to_pretraining_model --data_path path_to_dataset_for_evaluation  
~~~  

## TODO's  
add details on preparing estimated depth/region proposal  
upload pretrained models  
upload pre-generated depth/region proposal/datasets

