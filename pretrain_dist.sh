python pretrain_dist.py \
    --model_name=S3Rec \
    --gpu_id=3,4,5,6 \
    --dataset=us \
    --world_size=4 \
    --nproc=4 \
    --port=6006
    