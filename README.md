# AutoGUI

```
conda create -n autogui python=3.9
pip install torch==1.13.1+cu117 torchvision==0.14.1+cu117 torchaudio==0.13.1 --extra-index-url https://download.pytorch.org/whl/cu117
conda activate autogui

1. download mmocr checkpoints
cd playground

mkdir checkpoints
mkdir checkpoints/mmocr

1.1 down dbnet++ checkpoint, URL: https://drive.google.com/file/d/1r3B1xhkyKYcQ9SR7o9hw9zhNJinRiHD-/view?usp=share_link
mv db_swin_mix_pretrain.pth checkpoints/mmocr
1.2 down regnet checkpoint
wget -O checkpoints/mmocr/abinet_20e_st-an_mj_20221005_012617-ead8c139.pth https://download.openmmlab.com/mmocr/textrecog/abinet/abinet_20e_st-an_mj/abinet_20e_st-an_mj_20221005_012617-ead8c139.pth


2. download sam checkpoints
cd sam

mkdir checkpoints

2.1 download sam checkpoint, URL: https://dl.fbaipublicfiles.com/segment_anything/sam_vit_h_4b8939.pth
mv sam_vit_h_4b8939.pth checkpoints/


3. pip
pip install -r requirements.txt

4.run test
cd tools/
python mmocr_inference.py
python sam_inference.py
python scripts.py
```