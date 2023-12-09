# bash train_scripts/train_kd/train_sa.sh
CUDA_VISIBLE_DEVICES=3,5,6,7\
  nohup python3 -m torch.distributed.launch --nproc_per_node=4 --master_port='29500'\
  train_sa.py \
  --teacher-model deeplabv3 \
  --student-model psp \
  --teacher-backbone resnet101 \
  --student-backbone resnet18 \
  --lambda_cam 0.3 \
  --lambda_pam 0.8 \
  --sa_temperature 2 \
  --cwd_temperature 2 \
  --lambda-cwd-logit 3 \
  --data [your dataset path] \
  --save-dir [your directory path to store checkpoint files] \
  --teacher-pretrained [your teacher weights path] \
  --student-pretrained-base [your pretrained-backbone path] \
  > [your nohup output path] &