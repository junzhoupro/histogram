# !/bin/bash

torchrun --nnodes=1 --nproc_per_node=$(nvidia-smi --list-gpus | wc -l) src/train.py --outdir ./training-runs --img_path /disk3/proj_viton/jinx-synthetic/blender-files/jinx-synthetic --hdr_path /disk3/proj_viton/jinx-synthetic/hdrs \
         --epochs 200 --lr 5e-5 --batch_size 4 --num_workers 4 --log_every 100 --ckpt_every 5 \
         --out_activation log_softmax \
         --lambda_image 1 --lambda_lpips 0.2 --lambda_hist 0.1 --hist_loss kl