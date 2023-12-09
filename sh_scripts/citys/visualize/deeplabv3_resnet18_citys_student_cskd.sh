CUDA_VISIBLE_DEVICES=0,1 \
nohup python3 -m torch.distributed.launch --nproc_per_node=2 --master_port='29501'\
    val_visualize.py \
    --model deeplabv3 \
    --backbone resnet18 \
    --data [your dataset path] \
    --save-dir [your directory path to store segmentation maps] \
    --save-pred \
    --method CSKD \
    --pretrained [your pretrained-student path] \ 
    > [your nohup output path] &


