from trainer import Trainer
import os 
import argparse

parser = argparse.ArgumentParser(description="Implementation of FYA")
#data
parser.add_argument("--use_prepared_data", action="store_true",
                    help="use prepared mixing images")
parser.add_argument("--only_generate_data", action="store_true",
                    help="use prepared mixing images")
parser.add_argument("--data_path", type=str, required=True,
                    help="path to dataset repository")
parser.add_argument("--dataset", type=str, default="cs", choices=["cs", "kitti"],
                    help="cs or kitti")
parser.add_argument("--input_size", type=int, default=[384,768], nargs="+",
                    help="input size")
parser.add_argument("--mix_num", type=int, default=8,
                    help="number of image used in image mixing process")
parser.add_argument("--cp_times", type=int, default=[2,4], nargs="+",
                    help="copy-paste times for each region instance in the two rounds of image mixing process")
parser.add_argument("--cfg_file", type=str, default="./detectron2_model/Base-Panoptic-FPN.yaml",
                    help="detectron2 cfg file for network architecture")



#resume
parser.add_argument("--resume_model_path", type=str, default=None,
                    help="model path for resuming training")

#save
parser.add_argument("--model_name", type=str, default="model",
                    help="name of the model")
parser.add_argument("--save_path", type=str, default="./save_model",
                    help="model save path")
parser.add_argument("--save_frequency_epoch", type=int, default=1,
                    help="model save frequency")

#projector
parser.add_argument("--projector_layer_num", type=int, default=1,
                    help="model save frequency")
parser.add_argument("--projector_out_channels", type=int, default=128,
                    help="model save frequency")

#swav parameters
parser.add_argument("--temperature", type=float, default=0.1,
                    help="temperature of swav loss")
parser.add_argument("--epsilon", type=float, default=0.05,
                    help="epsilon in sinkhorn algorithm")
parser.add_argument("--num_prototypes", type=int, default=1000,
                    help="number of swav prototypes")
parser.add_argument("--sinkhorn_iterations", type=int, default=3,
                    help="number of iterations of skinhorm algorithm")

#training
parser.add_argument("--device", type=str, default="cuda:0",
                    help="training device")
parser.add_argument("--num_workers", type=int, default=6,
                    help="number of dataloader workers")
parser.add_argument("--batch_size", type=int, default=1,
                    help="number of training items per iteration")
parser.add_argument("--backbone_lr", type=float, default=1e-1,
                    help="backbone learning rate")
parser.add_argument("--sem_seg_head_lr", type=float, default=1e-1,
                    help="semantic segmentation head learning rate")
parser.add_argument("--projector_lr", type=float, default=1e-1,
                    help="projector learning rate")
parser.add_argument("--prototypes_lr", type=float, default=1e-1,
                    help="swav prototypes learning rate")
parser.add_argument("--final_lr", type=float, default=1e-5,
                    help="final learning in cosine decay schedule")
parser.add_argument("--warmup_epochs", type=int, default=2,
                    help="number of epochs for warmup")
parser.add_argument("--num_epochs", type=int, default=25,
                    help="number of total epochs")
parser.add_argument("--log_frequency", type=int, default=25,
                    help="frequency of printing training info")
#loss
parser.add_argument("--loss_pix_weight", type=float, default=0.5,
                    help="loss weight of pixel-level discrimination")
parser.add_argument("--loss_region_weight", type=float, default=0.5,
                    help="loss weight of region-level discrimination")
parser.add_argument("--pix_sample_per_img", type=int, default=4000,
                    help="sample number for pixel-level discrimination")
parser.add_argument("--region_sample_per_img", type=int, default=7200,
                    help="sample number for region-level discrimination")

if __name__ == "__main__":
    config = parser.parse_args()
    trainer = Trainer(config)
    trainer.train()

