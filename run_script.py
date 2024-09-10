import os
import argparse
from experiment_helpers.gpu_details import print_details
from accelerate import Accelerator
import time

parser=argparse.ArgumentParser()

parser.add_argument("--mixed_precision",type=str,default="no")
parser.add_argument("--project_name",type=str,default="evaluation-creative")
parser.add_argument("--dataset",type=str,default="jlbaker361/new_league_data_max_plus")
parser.add_argument("--pretrain_epochs",type=int,default=1)
parser.add_argument("--adversarial_epochs",type=int,default=10)
parser.add_argument("--load_pretrained_disc",action="store_true")
parser.add_argument("--output_dir",type=str,default="/scratch/jlb638/ddpogan/experiment")
parser.add_argument("--image_dir",type=str,default="/scratch/jlb638/ddpogan_images/experiment")



def main(args):
    accelerator=Accelerator(log_with="wandb",mixed_precision=args.mixed_precision)
    accelerator.init_trackers(project_name=args.project_name,config=vars(args))
    return

if __name__=='__main__':
    print_details()
    start=time.time()
    args=parser.parse_args()
    print(args)
    main(args)
    end=time.time()
    seconds=end-start
    hours=seconds/(60*60)
    print(f"successful generating:) time elapsed: {seconds} seconds = {hours} hours")
    print("all done!")