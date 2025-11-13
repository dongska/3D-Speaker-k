## Setting Up the Repository
Please run the following commands to set up the repository:
### Create a Conda Environment
```bash
conda create -n aum python=3.10.13
conda activate aum
```
### Setting Up CUDA and CuDNN
```bash
conda install nvidia/label/cuda-11.8.0::cuda-nvcc
conda install nvidia/label/cuda-11.8.0::cuda

Try: 
conda install anaconda::cudnn
Else:
conda install -c conda-forge cudnn
```
### Installing PyTorch and Other Dependencies
这里的requirements.txt已经针对3dspeaker做了合并
```bash
pip install torch==2.1.1 torchvision==0.16.1 torchaudio==2.1.1 --index-url https://download.pytorch.org/whl/cu118
pip install -r requirements.txt
```
### Installing Mamba Related Packages
```bash
pip install causal_conv1d==1.1.3.post1 mamba_ssm==1.1.3.post1
```
### Enabling Bidirectional SSM Processing
To integrate the modifications for supporting bidirectional processing, copy the `mamba_ssm` folder to the `site-packages` directory of the Python installation within the Conda environment. This folder is directly borrowed from the [ViM](https://github.com/hustvl/Vim) repository. 
```bash
cp -rf vim-mamba_ssm/mamba_ssm $CONDA_PREFIX/lib/python3.10/site-packages
```