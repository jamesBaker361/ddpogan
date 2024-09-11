import os
import argparse
from experiment_helpers.gpu_details import print_details
from accelerate import Accelerator
import time
from proto_gan_models import Discriminator
from datasets import load_dataset
import torch
from experiment_helpers.better_ddpo_pipeline import BetterDefaultDDPOStableDiffusionPipeline
from experiment_helpers.better_ddpo_trainer import BetterDDPOTrainer,get_image_sample_hook
from experiment_helpers.training import train_unet as train_unet_function
from trl import DDPOConfig
from PIL import Image
from torchvision import transforms
from static_globals import *
from proto_gan_diffaug import DiffAugment

parser=argparse.ArgumentParser()

parser.add_argument("--mixed_precision",type=str,default="no")
parser.add_argument("--project_name",type=str,default="evaluation-creative")
parser.add_argument("--dataset",type=str,default="jlbaker361/new_league_data_max_plus")
parser.add_argument("--pretrain_epochs",type=int,default=1)
parser.add_argument("--pretrain_steps_per_epoch",type=int,default=30)
parser.add_argument("--adversarial_epochs",type=int,default=10)
parser.add_argument("--discriminator_batch_size",type=int,default=8)
parser.add_argument("--load_pretrained_disc",action="store_true")
parser.add_argument("--output_dir",type=str,default="/scratch/jlb638/ddpogan/experiment")
parser.add_argument("--image_dir",type=str,default="/scratch/jlb638/ddpogan_images/experiment")
parser.add_argument("--pretrained_proto_gan",type=str,default="/scratch/jlb638/512_30000_proto_8")
parser.add_argument("--ddpo_lr",type=float,default=0.0001)
parser.add_argument("--train_gradient_accumulation_steps",type=int,default=8)
parser.add_argument("--num_inference_steps",type=int,default=30)
parser.add_argument("--ddpo_batch_size",type=int,default=1)
parser.add_argument("--pretrain_batch_size",type=int,default=8)
parser.add_argument("--samples_per_epoch",type=int,default=8)
parser.add_argument("--entity_name",type=str,default="league_of_legends_character")

evaluation_prompt_list=[
    " {} going for a walk ",
    " {} reading a book ",
    " {} playing guitar ",
    " {} baking cookies ",
    " {} in paris "
]


def main(args):
    accelerator=Accelerator(log_with="wandb",mixed_precision=args.mixed_precision)
    accelerator.init_trackers(project_name=args.project_name,config=vars(args))

    data=load_dataset(args.dataset,split="train")
    for row in data:
        break
    
    width,height=row["image"].size

    data=[row["image"] for row in data]

    proto_discriminator=Discriminator(64,3,height,1)

    if args.load_pretrained_disc:

        ckpt = torch.load(args.pretrained_proto_gan)
        proto_discriminator.load_state_dict(ckpt['d'])

    proto_discriminator=proto_discriminator.to(accelerator.device)

    transform_list = [
            transforms.Resize((width,height)),
            #transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
        ]
    composed_trans = transforms.Compose(transform_list)

    def get_proto_gan_score(image:Image.Image):
        tensor_img=composed_trans(image).unsqueeze(0).to(accelerator.device)
        pred, _, _,_, = proto_discriminator(tensor_img,"fake")
        return pred.mean().detach().cpu().numpy()
    
    pipeline=BetterDefaultDDPOStableDiffusionPipeline(
            False,
            False,
            True,
            False,
            use_lora=True,
            pretrained_model_name="CompVis/stable-diffusion-v1-4")

    config=DDPOConfig(
            train_learning_rate=args.ddpo_lr,
            num_epochs=1,
            train_gradient_accumulation_steps=args.train_gradient_accumulation_steps,
            sample_num_steps=args.num_inference_steps,
            sample_batch_size=args.ddpo_batch_size,
            train_batch_size=args.ddpo_batch_size,
            sample_num_batches_per_epoch=args.samples_per_epoch,
            mixed_precision=args.mixed_precision,
            tracker_project_name="ddpo-personalization",
            log_with="wandb",
            per_prompt_stat_tracking=args.project_name,
            accelerator_kwargs={
                #"project_dir":args.output_dir
            },)
    
    entity_name=args.entity_name.replace("_"," ")
    def prompt_fn():
        return entity_name,{}

    def reward_fn(images, prompts, epoch,prompt_metadata):
        return [get_proto_gan_score(image) for image in images],{}
                    

    image_samples_hook=get_image_sample_hook(args.image_dir)
    trainer = BetterDDPOTrainer(
        config,
        reward_fn,
        prompt_fn,
        pipeline,
        image_samples_hook,
        entity_name,
        height
    )
    print("len trainable parameters",len(pipeline.get_trainable_layers()))

    data=[composed_trans(image) for image in data]
    batched_data=[]
    for j in range(0,len(data),args.discriminator_batch_size):
        batched_data.append(data[j:j+args.discriminator_batch_size])
    batched_data=[torch.stack(batch) for batch in batched_data]

    if args.pretrain_epochs>0:
        #pretrain_image_list=[src_image] *pretrain_steps_per_epoch
        _pretrain_image_list=[]
        _pretrain_prompt_list=[]
        for x in range(args.pretrain_steps_per_epoch):
            if x%2==0:
                _pretrain_image_list.append(pretrain_image_list[x% len(pretrain_image_list)])
            else:
                _pretrain_image_list.append(pretrain_image_list[x% len(pretrain_image_list)].transpose(Image.FLIP_LEFT_RIGHT))
            _pretrain_prompt_list.append(entity_name)
        pretrain_prompt_list=_pretrain_prompt_list
        pretrain_image_list=_pretrain_image_list
        assert len(pretrain_image_list)==len(pretrain_prompt_list), f"error {len(pretrain_image_list)} != {len(pretrain_prompt_list)}"
        assert len(pretrain_image_list)==args.pretrain_steps_per_epoch, f"error {len(pretrain_image_list)} != {args.pretrain_steps_per_epoch}"
        pretrain_optimizer=trainer._setup_optimizer([p for p in pipeline.sd_pipeline.unet.parameters() if p.requires_grad])
        pipeline.sd_pipeline=train_unet_function(
            pipeline.sd_pipeline,
            args.pretrain_epochs,
            pretrain_image_list,
            pretrain_prompt_list,
            pretrain_optimizer,
            False,
            "prior",
            args.pretrain_batch_size,
            1.0,
            entity_name,
            trainer.accelerator,
            args.pretrain_steps_per_epoch,
            0.0,
            True
        )
        torch.cuda.empty_cache()
        trainer.accelerator.free_memory()

    pipeline.sd_pipeline.scheduler.alphas_cumprod=pipeline.sd_pipeline.scheduler.alphas_cumprod.to("cpu")
    policy = 'color,translation,cutout'
    print(f"acceleerate device {trainer.accelerator.device}")
    for e in range(args.adversarial_epochs):
        for _step,real_images in enumerate(batched_data):
            real_images=real_images.to(accelerator.device)

            real_images = DiffAugment(real_images, policy=policy)
            fake_images=[pipeline.sd_pipeline(entity_name,
                                                num_inference_steps=args.num_inference_steps,
                                                negative_prompt=NEGATIVE,
                                                width=width,
                                                height=height,
                                                safety_checker=None).images[0] for _ in range(len(args.discriminator_batch_size)) ]
            fake_images=[composed_trans(image) for image in fake_images]
            fake_images=torch.stack([DiffAugment(image) for image in fake_images]).to(accelerator.device)
            with accelerator.autocast():
                trainer.train(retain_graph=False,normalize_rewards=True)
        

    evaluation_image_list=[
        pipeline.sd_pipeline(evaluation_prompt.format(entity_name),
            num_inference_steps=args.num_inference_steps,
            negative_prompt=NEGATIVE,
            width=width,
            height=height,
            safety_checker=None).images[0] for evaluation_prompt in evaluation_prompt_list
    ]

    #evaluation: find most similar image in dataset and compare clip/vit/content/style similarities
    #evaluation: find average clip/style/similarities

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