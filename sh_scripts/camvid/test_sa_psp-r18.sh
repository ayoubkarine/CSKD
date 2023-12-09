# bash train_scripts/train_kd/train_sa.sh
CUDA_VISIBLE_DEVICES=0\
  nohup python3 -m torch.distributed.launch --nproc_per_node=1 --master_port='29504'\
  eval.py \
  --model psp \
  --backbone resnet18 \
  --dataset camvid \
  --data [your dataset path] \
  --save-dir [your directory path to store checkpoint files] \
  --pretrained [your pretrained-student path] \
  > [your nohup output path] &