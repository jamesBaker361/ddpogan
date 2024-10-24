import os
import argparse
from experiment_helpers.gpu_details import print_details
from accelerate import Accelerator
from accelerate.utils.random import set_seed
import time
from proto_gan_models import Discriminator,weights_init
from datasets import load_dataset
import torch
from experiment_helpers.better_ddpo_pipeline import BetterDefaultDDPOStableDiffusionPipeline
from experiment_helpers.better_ddpo_trainer import BetterDDPOTrainer,get_image_sample_hook
from experiment_helpers.training import train_unet, train_unet_single_prompt
from trl import DDPOConfig
from PIL import Image
from torchvision import transforms
from static_globals import *
from proto_gan_diffaug import DiffAugment
from peft.utils import get_peft_model_state_dict
from proto_gan_training import train_d
import wandb
from safetensors.torch import save_model
from clip_discriminator import ClipDiscriminator
import numpy as np
import random

parser=argparse.ArgumentParser()

parser.add_argument("--mixed_precision",type=str,default="no")
parser.add_argument("--style_list",nargs="*")
parser.add_argument("--project_name",type=str,default="ddpogan")
parser.add_argument("--dataset",type=str,default="jlbaker361/new_league_data_max_plus")
parser.add_argument("--hf_repo",type=str,default="jlbaker361/ddpo-gan")
parser.add_argument("--pretrain_epochs",type=int,default=1)
parser.add_argument("--adversarial_epochs",type=int,default=10)
parser.add_argument("--pretrain_steps_per_epoch",default=4,type=int)
parser.add_argument("--load_pretrained_disc",action="store_true")
parser.add_argument("--output_dir",type=str,default="/scratch/jlb638/ddpogan/experiment")
parser.add_argument("--image_dir",type=str,default="/scratch/jlb638/ddpogan_images/experiment")
parser.add_argument("--pretrained_proto_gan",type=str,default="/scratch/jlb638/512_30000_proto_8/all_20000.pth")
parser.add_argument("--ddpo_lr",type=float,default=0.0001)
parser.add_argument("--train_gradient_accumulation_steps",type=int,default=16)
parser.add_argument("--num_inference_steps",type=int,default=20)
parser.add_argument("--ddpo_batch_size",type=int,default=1)
parser.add_argument("--clip_classification_type",type=str, default=DROPOUT)
parser.add_argument("--pretrain_batch_size",type=int,default=8)
parser.add_argument("--samples_per_epoch",type=int,default=64)
parser.add_argument("--entity_name",type=str,default="league_of_legends_character")
parser.add_argument("--image_per_prompt",default=4,type=int)
parser.add_argument("--nlr",type=float,default=0.0002)
parser.add_argument("--nbeta1",type=float,default=0.5)
parser.add_argument("--image_size",type=int,default=512)
parser.add_argument("--save_interval",type=int,default=10)
parser.add_argument("--disc_batch_size",type=int,default=8)
parser.add_argument("--diffusion_start",type=int,default=0,help="how many adversarial epochs to wait before training ddpo")
parser.add_argument("--use_clip_discriminator",action="store_true")
parser.add_argument("--use_proto_discriminator",action="store_true")
parser.add_argument("--random_init",action="store_true")
parser.add_argument("--increasing_steps",action="store_true",help="whether to slwly increase the amount of steps used for training discriminator before adversarial training")

image_cache=[]

evaluation_prompt_list=[
    " {} going for a walk ",
    " {} reading a book ",
    " {} playing guitar ",
    " {} baking cookies ",
    " {} in paris "
]


def main(args):
    for d in [args.output_dir, args.image_dir]:
        os.makedirs(d,exist_ok=True)
    global image_cache
    
    if torch.cuda.is_available() and args.mixed_precision!="no":
        weight_dtype={"fp16":torch.float16}[args.mixed_precision]
    else:
        weight_dtype=torch.float32

    accelerator=Accelerator(log_with="wandb",mixed_precision=args.mixed_precision,
                            gradient_accumulation_steps=args.train_gradient_accumulation_steps)
    set_seed(42)

    
    accelerator.init_trackers(project_name=args.project_name,config=vars(args))

    data=load_dataset(args.dataset,split="train")
    if args.style_list is not None and len(args.style_list)>0:
        image_list=[row["image"].resize((args.image_size,args.image_size)) for row in data if row["style"] in args.style_list]
    else:
        image_list=[row["splash"].resize((args.image_size,args.image_size)) for row in data]
    print(f"selected {len(image_list)}/ {len(data)}")
    random.shuffle(image_list)
    if args.use_proto_discriminator:

        proto_discriminator=Discriminator(64,3,args.image_size,args.disc_batch_size)
        proto_discriminator.apply(weights_init)

        if args.load_pretrained_disc:

            try:
                ckpt = torch.load(args.pretrained_proto_gan)
            except RuntimeError:
                ckpt = torch.load(args.pretrained_proto_gan,map_location=torch.device('cpu'))
            proto_discriminator.load_state_dict(ckpt['d'])

        proto_discriminator=proto_discriminator.to(accelerator.device,weight_dtype=weight_dtype)

        optimizerD = torch.optim.Adam(proto_discriminator.parameters(), lr=args.nlr, betas=(args.nbeta1, 0.999))

        transform_list = [
                transforms.Resize((args.image_size,args.image_size)),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
            ]
        composed_trans = transforms.Compose(transform_list)

        def get_proto_gan_score(image:Image.Image):
            tensor_img=torch.stack([composed_trans(image).squeeze(0).to(accelerator.device) for _ in range(args.disc_batch_size)])
            print(tensor_img.size())
            pred, _, _,_, = proto_discriminator(tensor_img,"fake")
            print(pred)
            return -1.0 * pred.mean().detach().cpu().numpy().item()
        
        composed_data=[composed_trans(row["splash"]) for row in data]
        i=0
        while len(composed_data)%args.disc_batch_size !=0:
            composed_data.append(composed_data[i])
            i+=1
        batched_data=[]
        for j in range(0,len(composed_data),args.disc_batch_size):
            batched_data.append(composed_data[j:j+args.disc_batch_size])
        batched_data=[torch.stack(batch) for batch in batched_data]

        score_fn=get_proto_gan_score

    elif args.use_clip_discriminator:
        clip_disc=ClipDiscriminator(args.clip_classification_type,args.random_init,accelerator.device).to(weight_dtype)
        optimizerD = torch.optim.Adam(clip_disc.parameters(), lr=args.nlr, betas=(args.nbeta1, 0.999))

        def get_clip_score(image:Image.Image):
            pred=clip_disc(image)
            return pred.mean().detach().cpu().numpy().item()
        
        score_fn=get_clip_score
        i=0
        while len(image_list) %args.disc_batch_size !=0:
            image_list.append(image_list[i])
            i+=1
        batched_data=[]
        for j in range(0,len(image_list),args.disc_batch_size):
            batched_data.append(image_list[j:j+args.disc_batch_size])

        
    torch.cuda.empty_cache()
    accelerator.free_memory()


        
    
    
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
        global image_cache
        image_cache+=images
        rewards=[score_fn(image) for image in images]
        #print(rewards)
        return rewards,{}
                    

    image_samples_hook=get_image_sample_hook(args.image_dir)
    trainer = BetterDDPOTrainer(
        config,
        reward_fn,
        prompt_fn,
        pipeline,
        image_samples_hook,
        entity_name,
        args.image_size
    )
    print("len trainable parameters",len(pipeline.get_trainable_layers()))

    torch.cuda.empty_cache()
    accelerator.free_memory()


    if args.pretrain_epochs>0:
        #pretrain_image_list=[src_image] *pretrain_steps_per_epoch
        pretrain_optimizer=trainer._setup_optimizer([p for p in pipeline.sd_pipeline.unet.parameters() if p.requires_grad])
        pipeline.sd_pipeline=train_unet_single_prompt(
            pipeline.sd_pipeline,
            args.pretrain_epochs,
            image_list,
            entity_name,
            pretrain_optimizer,
            False,
            "prior",
            args.pretrain_batch_size,
            1.0,
            entity_name,
            trainer.accelerator,
            args.num_inference_steps,
            0.0,
            True
        )
        torch.cuda.empty_cache()
        trainer.accelerator.free_memory()

    pipeline.sd_pipeline.scheduler.alphas_cumprod=pipeline.sd_pipeline.scheduler.alphas_cumprod.to("cpu")
    policy = 'color,translation,cutout'
    print(f"acceleerate device {trainer.accelerator.device}")
    for e in range(1,args.adversarial_epochs+1):
        random.shuffle(batched_data)
        start=time.time()
        err_dr_list=[]
        fake_err_dr_list=[]
        for _step,real_images in enumerate(batched_data):
            #with accelerator.accumulate(pipeline.sd_pipeline.unet,clip_disc): #change this for proto_disc
            optimizerD.zero_grad()
            if _step%args.train_gradient_accumulation_steps==0:
                if e>=args.diffusion_start:
                    image_cache=[]
                    with accelerator.autocast():
                        trainer.train(retain_graph=False,normalize_rewards=True)
                    
                else:
                    image_cache=[]
                    steps=args.num_inference_steps
                    if args.increasing_steps:
                        steps = int(e * args.num_inference_steps/ args.diffusion_start)
                    steps=max(steps,2)
                    for i in range(args.disc_batch_size*args.train_gradient_accumulation_steps):
                        prompt,_=prompt_fn()
                        image_cache.append(pipeline.sd_pipeline(prompt,
                            height=args.image_size,
                            width=args.image_size,num_inference_steps=steps,
                            negative_prompt=NEGATIVE,safety_checker=None).images[0])
                fake_images=image_cache

            torch.cuda.empty_cache()
            trainer.accelerator.free_memory()

            if args.use_proto_discriminator:
                real_images=real_images.to(accelerator.device,dtype=weight_dtype)

                real_images = DiffAugment(real_images, policy=policy)
                print("real",real_images.size())
                '''fake_images=[pipeline.sd_pipeline(entity_name,
                                                    num_inference_steps=args.num_inference_steps,
                                                    negative_prompt=NEGATIVE,
                                                    width=width,
                                                    height=height,
                                                    safety_checker=None).images[0] for _ in range(len(args.disc_batch_size)) ]'''
                
                
                fake_images=[composed_trans(image) for image in fake_images]
                fake_images=torch.stack(fake_images).to(accelerator.device,dtype=weight_dtype)
                fake_images=DiffAugment(fake_images,policy=policy)
                print(fake_images.size())
                err_dr, rec_img_all, rec_img_small, rec_img_part = train_d(proto_discriminator, real_images, label="real")
                fake_err_dr=train_d(proto_discriminator, fake_images, label="fake")

                
            
            elif args.use_clip_discriminator:
                with accelerator.accumulate(clip_disc):
                    composed_trans=transforms.Compose([transforms.RandomHorizontalFlip(), transforms.RandomCrop((args.image_size*7)//8)])
                    real_images=[composed_trans(ri) for ri in real_images]
                    predictions_real=clip_disc(real_images)
                    print("pred real", predictions_real)
                    real_labels=0.95 * torch.ones(predictions_real.size()).to(accelerator.device,dtype=weight_dtype)+torch.normal(0,0.05,predictions_real.size()).to(accelerator.device,dtype=weight_dtype)

                    err_dr=torch.nn.functional.mse_loss(real_labels, predictions_real)
                    #err_dr.backward()
                    accelerator.backward(err_dr)

                    index=_step%args.train_gradient_accumulation_steps - (args.train_gradient_accumulation_steps%args.disc_batch_size)
                    predictions_fake=clip_disc(fake_images[index:index+args.disc_batch_size])
                    #print("pred fake", predictions_fake)
                    #print('len image cache',len(image_cache),'index ',index, 'index+args.disc_batch_size ',index+args.disc_batch_size)
                    fake_labels=torch.zeros(predictions_fake.size()).to(accelerator.device,dtype=weight_dtype)+ 0.1*torch.rand(predictions_real.size()).to(accelerator.device,dtype=weight_dtype)
                    fake_err_dr=torch.nn.functional.mse_loss(fake_labels, predictions_fake)
                    #fake_err_dr.backward()
                    accelerator.backward(fake_err_dr)

            err_dr_list.append(err_dr.detach().cpu().numpy())
            fake_err_dr_list.append(fake_err_dr.detach().cpu().numpy())
            optimizerD.step()


        end=time.time()
        print(f"epoch {e} ended after {end-start} seconds = {(end-start)/3600} hours")
        metrics={
            "err_dr":np.mean(err_dr_list),
            "fake_err_dr":np.mean(fake_err_dr_list),
        }
        for k,v in metrics.items():
            print("\t",k,v)
        accelerator.log(metrics)
        if e % args.save_interval == 0 or e == args.adversarial_epochs:
            if args.use_proto_discriminator:
                torch.save({'d':proto_discriminator.state_dict(),
                            'opt_d': optimizerD.state_dict()}, args.output_dir+'/all_%d.pth'%e)
            elif args.use_clip_discriminator:
                torch.save({'d':clip_disc.state_dict(),
                            'opt_d': optimizerD.state_dict()}, args.output_dir+'/all_%d.pth'%e)
            unet_lora_layers = get_peft_model_state_dict(pipeline.sd_pipeline.unet)
            pipeline.sd_pipeline.save_lora_weights(args.output_dir,unet_lora_layers)
            pipeline.sd_pipeline.save_pretrained(args.output_dir,push_to_hub=True, repo_id=args.hf_repo)
        
            




        

    evaluation_image_list=[]
    for j,evaluation_prompt in enumerate(evaluation_prompt_list):
        for n in range(args.image_per_prompt):
            generator=torch.Generator()
            generator=generator.manual_seed(n)
            image=pipeline.sd_pipeline(evaluation_prompt.format(entity_name),
                num_inference_steps=args.num_inference_steps,
                negative_prompt=NEGATIVE,
                width=args.image_size,
                height=args.image_size,
                generator=generator,
                safety_checker=None).images[0]
            evaluation_image_list.append(image)
            try:
                accelerator.log({
                    f"{j}":wandb.Image(image)
                })
            except:
                temp="temp.png"
                image.save(temp)
                accelerator.log({
                    f"{j}":wandb.Image(temp)
                })

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