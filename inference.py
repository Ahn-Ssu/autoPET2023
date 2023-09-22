from email.header import make_header
import sys
from monai.transforms import (
    Compose,
    LoadImaged,
    Orientationd,
    ScaleIntensityRanged,
    EnsureTyped,
    CropForeground
)
from monai.inferers import sliding_window_inference
from monai.data import list_data_collate, decollate_batch, DataLoader, Dataset
import torch
import pytorch_lightning

import os
import glob
import numpy as np

import nibabel as nib

from model import UNet_middleF


class Net(pytorch_lightning.LightningModule):
    def __init__(self):
        super().__init__()
        self.model = UNet_middleF(
                            input_dim=2,
                            out_dim=2,
                            hidden_dims=[32,32,64,128,256], # 16 32 32 64 128 is default setting of Monai
                            spatial_dim=3,
                            dropout_p=0.,
                            use_MS=False
                        )
        
        self.all_key = ['ct','pet']
        self.transform = Compose([
            LoadImaged(keys=self.all_key, ensure_channel_first=True, image_only=True),
            EnsureTyped(keys=self.all_key, track_meta=False), # [26 26  0] [373 373 326] torch.Size([1, 1, 400, 400, 326])
            Orientationd(keys=self.all_key, axcodes='RAS'),
            ScaleIntensityRanged(keys='ct',
                                    a_min=-1000, a_max=1000,
                                    b_min=0, b_max=1, clip=True),
            ScaleIntensityRanged(keys='pet',
                                    a_min=0, a_max=40,
                                    b_min=0, b_max=1, clip=False),
        ]
        )

    def forward(self, x):
        return self.model(x)

    def prepare_data(self, data_dir):
        # set up the correct data path
        images_pt = sorted(glob.glob(os.path.join(data_dir, "SUV*")))
        images_ct = sorted(glob.glob(os.path.join(data_dir, "CTres*")))
       
        val_files = [
            {"pet": image_name_pt, "ct": image_name_ct}
            for image_name_pt, image_name_ct in zip(images_pt, images_ct)
        ]

        self.val_ds = Dataset(data=val_files, transform=self.transform)

    def val_dataloader(self):
        val_loader = DataLoader(
            # we have to use the batch_size as 1 for coreground cropping
            self.val_ds, batch_size=1, num_workers=0, collate_fn = list_data_collate)
        return val_loader
    
    def load_weights(self, ckpt_path):
        # from collections import OrderedDict
        ckpt = torch.load(ckpt_path, map_location='cpu')
        # new_state_dict = OrderedDict()
        # for n, v in ckpt.items():
        #     name = n.replace("model.","")
        #     new_state_dict[name] = v
        self.model.load_state_dict(ckpt)


def segment_PETCT(ckpt_path, data_dir, export_dir):
    print("starting")

    net = Net()
    net.load_weights(ckpt_path=ckpt_path)
    net.eval()

    device = torch.device('cuda:0')
    net.to(device)
    net.prepare_data(data_dir)
    
    cropper = CropForeground()

    with torch.no_grad():
        for _, val_data in enumerate(net.val_dataloader()):
            pet = val_data['pet'][0]
            *_, H, W, D = val_data['pet'].shape # Bz, C, H, W, D -> (Bz, C), ...
            
            # background crop
            box_start, box_end = cropper.compute_bounding_box(img=pet)
            w_start, h_start, d_start = box_start
            w_end, h_end, d_end = box_end
            # print(box_start, box_end, val_data['pet'].shape)
            zeros = np.zeros((H,W,D), dtype=np.uint8)
            
            # inference 
            ct, pet = val_data['ct'], val_data['pet']
            ct = ct[..., w_start:w_end, h_start:h_end, d_start:d_end]
            pet = pet[..., w_start:w_end, h_start:h_end, d_start:d_end]
            image = torch.concat([ct,pet],dim=1).to(device) # Bz, C, H, W, D -- dim=1
            
            mask_out = sliding_window_inference(
                                    inputs=image,
                                    roi_size=(192, 192, 192),
                                    sw_batch_size=1,
                                    predictor=net.model,
                                    overlap=0.5,
                                    mode='constant')
            
            mask_out = torch.nn.functional.softmax(mask_out, dim=1)
            mask_out = torch.argmax(mask_out, dim=1)
            mask_out = torch.where(mask_out > 0.5, 1. , 0)
            mask_out = mask_out.detach().cpu().numpy().squeeze()
            mask_out = mask_out.astype(np.uint8)
            

            
            zeros[w_start:w_end, h_start:h_end, d_start:d_end] = mask_out
            print("done inference")

            
            PT = nib.load(os.path.join(data_dir,"SUV.nii.gz"))  #needs to be loaded to recover nifti header and export mask
            pet_affine = PT.affine
            mask_export = nib.Nifti1Image(zeros, pet_affine)
            print(os.path.join(export_dir, "PRED.nii.gz"))

            nib.save(mask_export, os.path.join(export_dir, "PRED.nii.gz"))
            print("done writing")


def run_inference(ckpt_path='/opt/algorithm/DiWA3.ckpt', data_dir='/opt/algorithm/', export_dir='/output/'):
    segment_PETCT(ckpt_path, data_dir, export_dir)


if __name__ == '__main__':
    run_inference()

