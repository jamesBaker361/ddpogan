import os
import argparse
from experiment_helpers.gpu_details import print_details
from accelerate import Accelerator
import time
from proto_gan_models import weights_init, Discriminator, Generator
from proto_gan_operations import copy_G_params, load_params, get_dir,ImageFolder, InfiniteSamplerWrapper
from proto_gan_diffaug import DiffAugment

from tqdm import tqdm
import torch
import torch.nn.functional as F
from torchvision.utils import save_image
from torchvision import transforms
import torch.optim as optim
from proto_gan_lpips.lpips_utils import PerceptualLoss
percept=PerceptualLoss(model='net-lin', net='vgg', use_gpu=torch.cuda.is_available())
from datasets import load_dataset
import numpy as np
import wandb
from huggingface_hub import HfApi
api = HfApi()
from PIL import Image,UnidentifiedImageError
from experiment_helpers.checkpoint import find_latest_checkpoint
import re

parser=argparse.ArgumentParser()

parser.add_argument("--mixed_precision",type=str,default="no")
parser.add_argument("--project_name",type=str,default="proto-gan")
parser.add_argument("--size",type=int,default=512)
parser.add_argument("--epochs",default=1000,type=int)
parser.add_argument("--ngf",default=64, type=int)
parser.add_argument("--ndf",type=int,default=64)
parser.add_argument("--nz",type=int,default=256)
parser.add_argument("--batch_size",type=int,default=8)
parser.add_argument("--nlr",type=float,default=0.0002)
parser.add_argument("--nbeta1",type=float,default=0.5)
parser.add_argument("--checkpoint",type=str, default="None")
parser.add_argument("--hf_dataset",type=str,default="jlbaker361/new_league_data_max_plus")
parser.add_argument("--save_interval",type=int,default=1000)
parser.add_argument("--image_dir",type=str,default="/scratch/jlb638/proto_images/")
parser.add_argument("--checkpoint_dir",type=str,default="/scratch/jlb638/proto_checkpoints/")
parser.add_argument("--repo_id",type=str,default="jlbaker361/proto-gan-512")
parser.add_argument("--test_data",action="store_true")
parser.add_argument("--load_from",action="store_true")

def crop_image_by_part(image, part):
    hw = image.shape[2]//2
    if part==0:
        return image[:,:,:hw,:hw]
    if part==1:
        return image[:,:,:hw,hw:]
    if part==2:
        return image[:,:,hw:,:hw]
    if part==3:
        return image[:,:,hw:,hw:]

def train_d(net, data, label="real"):
    """Train function of discriminator"""
    if label=="real":
        pred, [rec_all, rec_small, rec_part], part, _, _, = net(data, label)
        #print("pred size",pred.size())
        err = F.relu(torch.rand_like(pred) * 0.2 + 0.8 - pred).mean() + \
            percept( rec_all, F.interpolate(data, rec_all.shape[2]) ).sum() +\
            percept( rec_small, F.interpolate(data, rec_small.shape[2]) ).sum() +\
            percept( rec_part, F.interpolate(crop_image_by_part(data, part), rec_part.shape[2]) ).sum()
        err.backward()
        return pred.mean().item(), rec_all, rec_small, rec_part
    else:
        pred, _, _,_, = net(data, label)
        err = F.relu(torch.rand_like(pred) * 0.2 + 0.8 + pred).mean()
        # err = F.binary_cross_entropy_with_logits( pred, torch.zeros_like(pred)).mean()
        err.backward()
        return pred.mean().item()

def main(args):
    for folder in [args.image_dir, args.checkpoint_dir]:
        os.makedirs(folder,exist_ok=True)

    api.create_repo(args.repo_id,exist_ok=True)
    accelerator=Accelerator(log_with="wandb",mixed_precision=args.mixed_precision)
    accelerator.init_trackers(project_name=args.project_name,config=vars(args))
    transform_list = [
            transforms.Resize((int(args.size),int(args.size))),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
        ]
    trans = transforms.Compose(transform_list)
    device=accelerator.device
    netG = Generator(ngf=args.ngf, nz=args.nz, im_size=args.size)
    total_params_g = sum(p.numel() for p in netG.parameters())
    netG.apply(weights_init)

    netD = Discriminator(ndf=args.ndf, im_size=args.size,batch_size=args.batch_size)
    total_params_d = sum(p.numel() for p in netD.parameters())
    netD.apply(weights_init)
    total_params = total_params_d + total_params_g
    print("total_param",total_params)
    print('total_params_d',total_params_d)
    print("total_params_g",total_params_g)

    netG.to(device)
    netD.to(device)

    avg_param_G = copy_G_params(netG)

    fixed_noise = torch.FloatTensor(args.batch_size, args.nz).normal_(0, 1).to(device)

    optimizerG = optim.Adam(netG.parameters(), lr=args.nlr, betas=(args.nbeta1, 0.999))
    optimizerD = optim.Adam(netD.parameters(), lr=args.nlr, betas=(args.nbeta1, 0.999))
    current_epoch=0
    if args.load_from:
        pattern=re.compile(r"all_(\d+)\.pt")
        path,current_epoch=find_latest_checkpoint(args.checkpoint_dir,pattern)
        if path is not None:
            path=os.path.join(args.checkpoint_dir,path)
            ckpt = torch.load(path)
            print(f"loading from {path}")
            netG.load_state_dict(ckpt['g'])
            netD.load_state_dict(ckpt['d'])
            avg_param_G = ckpt['g_ema']
            optimizerG.load_state_dict(ckpt['opt_g'])
            optimizerD.load_state_dict(ckpt['opt_d'])
            #current_epoch = int(args.checkpoint.split('_')[-1].split('.')[0])
            del ckpt

    data=[row["splash"] for row in load_dataset(args.hf_dataset,split="train")]
    if args.test_data:
        data=[Image.open("boot.jpg") for _ in range(32)]
    i=0
    while len(data) %args.batch_size!=0:
        data.append(data[i])
        i+=1
    policy = 'color,translation,cutout'
    data=[trans(image) for image in data]
    batched_data=[]
    for j in range(0,len(data),args.batch_size):
        batched_data.append(data[j:j+args.batch_size])
    batched_data=[torch.stack(batch) for batch in batched_data]
    step=0
    total_steps=len(batched_data)*args.epochs
    for e in tqdm(range(current_epoch, args.epochs+1)):
        start=time.time()
        err_dr_list=[]
        fake_err_dr_list=[]
        err_g_list=[]
        pred_g_list=[]
        for _step,real_images in enumerate(batched_data):
            real_images=real_images.to(device)
            noise = torch.Tensor(args.batch_size, args.nz).normal_(0, 1).to(device)

            fake_images = netG(noise)

            real_images = DiffAugment(real_images, policy=policy)
            fake_images = [DiffAugment(fake, policy=policy) for fake in fake_images]

            ## 2. train Discriminator
            netD.zero_grad()

            err_dr, rec_img_all, rec_img_small, rec_img_part = train_d(netD, real_images, label="real")
            fake_err_dr=train_d(netD, [fi.detach() for fi in fake_images], label="fake")

            #err_dr_list.append(err_dr.detach().cpu().numpy())
            err_dr_list.append(err_dr)
            #fake_err_dr_list.append(fake_err_dr.detach().cpu().numpy())
            fake_err_dr_list.append(fake_err_dr)
            optimizerD.step()
            ## 3. train Generator

            netG.zero_grad()
            pred, [rec_all, rec_small, rec_part], part, feat_R, feat_mean_R = netD(real_images, label="real")
            current_feat_mean_real = feat_mean_R
            # aggregate the features along training time 
            if step == 0:
                total_feat_mean_real = current_feat_mean_real
            else:
                total_feat_mean_real = (step * total_feat_mean_real +  current_feat_mean_real) / (step + 1)
            total_feat_mean_real = total_feat_mean_real.detach()

            pred_g, feat_F, feat_mean_F, feat_var_F= netD(fake_images, "fake")
            pred_g_list.append(pred_g.detach().cpu().numpy())
            #pred_g_list.append(pred_g)
            #sig_loss = torch.mean(np.square(noise_zsig - 1))
            #err_g = -F.binary_cross_entropy_with_logits(pred_g, torch.zeros_like(pred_g))
            matching_loss =  feat_F - feat_R
            proto_loss = feat_mean_F - total_feat_mean_real
            var_loss = feat_var_F
            # optimize the generator with ptrototype, feature matching and variance loss
            err_g = -pred_g.mean() + (step / total_steps) * matching_loss.mean() + (step / total_steps) * proto_loss.mean() - 2 * var_loss.mean()
            err_g_list.append(err_g.detach().cpu().numpy())

            err_g.backward()
            optimizerG.step()
        end=time.time()
        print(f"epoch {e} elapsed {end-start} seconds")
        metrics={
            "err_dr":np.mean(err_dr_list),
            "fake_err_dr":np.mean(fake_err_dr_list),
            "err_g":np.mean(err_g_list),
            "pred_g":np.mean(pred_g_list)
        }
        for k,v in metrics.items():
            print("\t",k,v)
        accelerator.log(metrics)
        fixed_images=[fake.add(1).mul(0.5) for fake in netG(fixed_noise)]
        for index,fake in enumerate(fixed_images):
            path=f"{args.image_dir}{index}.jpg"
            save_image(fake,path)
            try:
                accelerator.log({
                    f"image_{index}":wandb.Image(path)
                })
            except UnidentifiedImageError:
                print(f"couldnt find {path} during epoch {e}")
        if e % args.save_interval == 0 or e == args.epochs:
            backup_para = copy_G_params(netG)
            load_params(netG, avg_param_G)
            torch.save({'g':netG.state_dict(),'d':netD.state_dict()}, args.checkpoint_dir+'/%d.pth'%e)
            load_params(netG, backup_para)
            torch.save({'g':netG.state_dict(),
                        'd':netD.state_dict(),
                        'g_ema': avg_param_G,
                        'opt_g': optimizerG.state_dict(),
                        'opt_d': optimizerD.state_dict()}, args.checkpoint_dir+'/all_%d.pth'%e)
            api.upload_folder(repo_id=args.repo_id,
                              repo_type="model",
                              folder_path=args.checkpoint_dir)
            





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