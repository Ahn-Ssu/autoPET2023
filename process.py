import SimpleITK
import numpy as np
import torch

import inference
import os
import shutil


from monai.inferers import sliding_window_inference
from monai.transforms import CropForeground

class Unet_baseline():  # SegmentationAlgorithm is not inherited in this class anymore
    def __init__(self):
        """
        Write your own input validators here
        Initialize your model etc.
        """
        self.input_path = '/input/'  # according to the specified grand-challenge interfaces
        self.output_path = '/output/images/automated-petct-lesion-segmentation/'  # according to the specified grand-challenge interfaces
        self.nii_path = '/opt/algorithm/'  # where to store the nii files
        self.ckpt_path = '/opt/algorithm/DiWA3.ckpt'
        if not os.path.exists(self.output_path):
            os.makedirs(self.output_path)
        
        self.net = inference.Net()


    def convert_mha_to_nii(self, mha_input_path, nii_out_path):
        img = SimpleITK.ReadImage(mha_input_path)
        SimpleITK.WriteImage(img, nii_out_path, True)

    def convert_nii_to_mha(self, nii_input_path, mha_out_path):
        img = SimpleITK.ReadImage(nii_input_path)
        SimpleITK.WriteImage(img, mha_out_path, True)

    def check_gpu(self):
        """
        Check if GPU is available
        """
        print('Checking GPU availability')
        is_available = torch.cuda.is_available()
        print('Available: ' + str(is_available))
        print(f'Device count: {torch.cuda.device_count()}')
        if is_available:
            print(f'Current device: {torch.cuda.current_device()}')
            print('Device name: ' + torch.cuda.get_device_name(0))
            print('Device memory: ' + str(torch.cuda.get_device_properties(0).total_memory))

    def load_inputs(self):
        """
        Read from /input/
        Check https://grand-challenge.org/algorithms/interfaces/
        """
        ct_mha = os.listdir(os.path.join(self.input_path, 'images/ct/'))[0]
        pet_mha = os.listdir(os.path.join(self.input_path, 'images/pet/'))[0]
        uuid = os.path.splitext(ct_mha)[0]

        self.convert_mha_to_nii(os.path.join(self.input_path, 'images/pet/', pet_mha),
                                os.path.join(self.nii_path, 'SUV.nii.gz'))
        self.convert_mha_to_nii(os.path.join(self.input_path, 'images/ct/', ct_mha),
                                os.path.join(self.nii_path, 'CTres.nii.gz'))
        return uuid

    def write_outputs(self, uuid):
        """
        Write to /output/
        Check https://grand-challenge.org/algorithms/interfaces/
        """
        os.makedirs(os.path.dirname(self.output_path), exist_ok=True)
        self.convert_nii_to_mha(os.path.join(self.output_path, "PRED.nii.gz"), os.path.join(self.output_path, uuid + ".mha"))
        print('Output written to: ' + os.path.join(self.output_path, uuid + ".mha"))
    
    def predict(self, inputs):
        """
        Your algorithm goes here
        """        
        self.net.load_weights(self.ckpt_path)
        self.net.eval()
        
        device = torch.device('cuda:0')
        cropper = CropForeground()
        
        with torch.no_grad():
            pet = inputs['pet'][0]
            *_, H, W, D = inputs['pet'].shape # Bz, C, H, W, D -> (Bz, C), ...
            
            # background crop
            box_start, box_end = cropper.compute_bounding_box(img=pet)
            w_start, h_start, d_start = box_start
            w_end, h_end, d_end = box_end
            outputs = np.zeros((H,W,D), dtype=np.uint8)
            
            # inference 
            ct, pet = inputs['ct'], inputs['pet']
            ct = ct[..., w_start:w_end, h_start:h_end, d_start:d_end]
            pet = pet[..., w_start:w_end, h_start:h_end, d_start:d_end]
            image = torch.concat([ct,pet],dim=1).to(device) # Bz, C, H, W, D -- dim=1
            
            mask_out = sliding_window_inference(
                                    inputs=image,
                                    roi_size=(128, 128, 128),
                                    sw_batch_size=2,
                                    predictor=self.net.model,
                                    overlap=0.5,
                                    mode='constant')
            
            mask_out = torch.nn.functional.softmax(mask_out, dim=1)
            mask_out = torch.argmax(mask_out, dim=1)
            mask_out = torch.where(mask_out > 0.5, 1. , 0)
            mask_out = mask_out.detach().cpu().numpy().squeeze()
            mask_out = mask_out.astype(np.uint8)
            
            outputs[w_start:w_end, h_start:h_end, d_start:d_end] = mask_out
        
        
        
        return outputs

    def process(self):
        """
        Read inputs from /input, process with your algorithm and write to /output
        """
        self.check_gpu()
        print('Start processing')
        uuid = self.load_inputs()
        print('Start prediction')
        inference.run_inference(self.ckpt_path, self.nii_path, self.output_path)
        print('Start output writing')
        self.write_outputs(uuid)


if __name__ == "__main__":
    Unet_baseline().process()
