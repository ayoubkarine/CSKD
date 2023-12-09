CUDA_VISIBLE_DEVICES=2,3\
  nohup python3 -m torch.distributed.launch --nproc_per_node=2 --master_port='29504'\
  train_sa.py \
  --teacher-model deeplabv3 \
  --student-model deeplabv3 \
  --teacher-backbone resnet101 \
  --student-backbone resnet18 \
  --dataset camvid \
  --crop-size 360 360 \
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