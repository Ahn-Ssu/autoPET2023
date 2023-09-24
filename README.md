# Domain-Adaptive PET/CT Tumor Lesion Segmentation Networks through Effective Training Methods
AutoPET 2023 challenge submission repository.

## How to use
- Prepare the checkpoints
  - Download the checkpoints from [Google Drive](https://drive.google.com/drive/folders/1_5yYclCru3PymlCOLcZaardurSTM0y5k?usp=sharing) or [Cloud](https://drive.google.com/drive/folders/1_5yYclCru3PymlCOLcZaardurSTM0y5k?usp=sharing).
  - unzip `ckpt.zip` file.
  - locate the weight files under the root of this repository.
- Make the docker container
  - Run `build.sh`. If you want to run the evaluation code, run `test.sh` with given samples from under `test` of [autoPET repo](https://github.com/lab-midas/autoPET)
