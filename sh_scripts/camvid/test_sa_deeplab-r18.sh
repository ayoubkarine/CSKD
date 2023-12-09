CUDA_VISIBLE_DEVICES=3\
  nohup python3 -m torch.distributed.launch --nproc_per_node=1 --master_port='29500'\
  eval.py \
  --model deeplabv3 \
  --backbone resnet18 \
  --dataset camvid \
  --data [your dataset path] \
  --save-dir [your directory path to store checkpoint files] \
  --pretrained [your pretrained-student path] \
  > [your nohup output path] &