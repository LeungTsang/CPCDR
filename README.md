# MSc-Project：Leverage Depth and Copy-Paste to Boost Contrastive Learning for Urban-Scene Segmentation
![Pipeline](https://github.com/LeungTsang/MSc-Project/raw/main/fig/pipeline.png)

![Visualization](https://github.com/LeungTsang/MSc-Project/raw/main/fig/fig1.gif)

Thesis: http://resolver.tudelft.nl/uuid:3c7ba0bd-b467-4e1d-91df-c3ecceeec344

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
~~~
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
##simple usage  
### Training  
~~~  
python run.py --data_path path_to_dataset_for_training  
### Evaluation  
~~~  
python eval_unsupervised.py --model_path path_to_pretraining_model --data_path path_to_dataset_for_evaluation  

