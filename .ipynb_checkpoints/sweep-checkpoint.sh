#!/bin/bash

# CUDA_VISIBLE_DEVICES=3 python train.py torch/cifar100 --dataset torch/cifar100 -b=128 --img-size=224 --epochs=50 --color-jitter=0 --amp --lr=1e-3 --sched='cosine' --model-ema --model-ema-decay=0.995 --reprob=0.5 --smoothing=0.1 --min-lr=1e-8 --warmup-epochs=3 --train-interpolation=bilinear --aa=v0 --model=resnet50 --pretrained --num-classes=100 --opt=sgd --weight-decay=1e-4 --log-wandb --dataset-download --experiment "sweep1"  --name  "sgd lr=1e-3 base" &


# CUDA_VISIBLE_DEVICES=4 python train.py torch/cifar100 --dataset torch/cifar100 -b=128 --img-size=224 --epochs=50 --color-jitter=0 --amp --lr=1e-4 --sched='cosine' --model-ema --model-ema-decay=0.995 --reprob=0.5 --smoothing=0.1 --min-lr=1e-8 --warmup-epochs=3 --train-interpolation=bilinear --aa=v0 --model=resnet50 --pretrained --num-classes=100 --opt=sgd --weight-decay=1e-4 --log-wandb --dataset-download --experiment "sweep1"  --name  "sgd lr=1e-4 base" &



# CUDA_VISIBLE_DEVICES=5 python train.py torch/cifar100 --dataset torch/cifar100 -b=128 --img-size=224 --epochs=50 --color-jitter=0 --amp --lr=1e-3 --sched='cosine' --model-ema --model-ema-decay=0.995 --reprob=0.5 --smoothing=0.1 --min-lr=1e-8 --warmup-epochs=3 --train-interpolation=bilinear --aa=v0 --model=resnet50 --pretrained --num-classes=100 --opt=adam --weight-decay=1e-4 --log-wandb --dataset-download --experiment "sweep1"  --name  "adam lr=1e-3 base" &


# CUDA_VISIBLE_DEVICES=6 python train.py torch/cifar100 --dataset torch/cifar100 -b=128 --img-size=224 --epochs=50 --color-jitter=0 --amp --lr=1e-4 --sched='cosine' --model-ema --model-ema-decay=0.995 --reprob=0.5 --smoothing=0.1 --min-lr=1e-8 --warmup-epochs=3 --train-interpolation=bilinear --aa=v0 --model=resnet50 --pretrained --num-classes=100 --opt=adam --weight-decay=1e-4 --log-wandb --dataset-download --experiment "sweep1"  --name  "adam lr=1e-4 base" &






# CUDA_VISIBLE_DEVICES=3 python train.py torch/cifar100 --dataset torch/cifar100 -b=128 --img-size=224 --epochs=50 --color-jitter=0 --amp --lr=1e-3 --sched='cosine' --model-ema --model-ema-decay=0.995 --reprob=0.5 --smoothing=0.1 --min-lr=1e-8 --warmup-epochs=3 --train-interpolation=bilinear --aa=v0 --model=resnet50 --pretrained --num-classes=100 --opt=sgd --weight-decay=1e-4 --log-wandb --dataset-download --experiment "sweep1" --dual --name  "sgd lr=1e-3 dual" &


# CUDA_VISIBLE_DEVICES=4 python train.py torch/cifar100 --dataset torch/cifar100 -b=128 --img-size=224 --epochs=50 --color-jitter=0 --amp --lr=1e-4 --sched='cosine' --model-ema --model-ema-decay=0.995 --reprob=0.5 --smoothing=0.1 --min-lr=1e-8 --warmup-epochs=3 --train-interpolation=bilinear --aa=v0 --model=resnet50 --pretrained --num-classes=100 --opt=sgd --weight-decay=1e-4 --log-wandb --dataset-download --experiment "sweep1" --dual --name  "sgd lr=1e-4 dual" &



# CUDA_VISIBLE_DEVICES=5 python train.py torch/cifar100 --dataset torch/cifar100 -b=128 --img-size=224 --epochs=50 --color-jitter=0 --amp --lr=1e-3 --sched='cosine' --model-ema --model-ema-decay=0.995 --reprob=0.5 --smoothing=0.1 --min-lr=1e-8 --warmup-epochs=3 --train-interpolation=bilinear --aa=v0 --model=resnet50 --pretrained --num-classes=100 --opt=adam --weight-decay=1e-4 --log-wandb --dataset-download --experiment "sweep1" --dual --name  "adam lr=1e-3 dual" &


# CUDA_VISIBLE_DEVICES=6 python train.py torch/cifar100 --dataset torch/cifar100 -b=128 --img-size=224 --epochs=50 --color-jitter=0 --amp --lr=1e-4 --sched='cosine' --model-ema --model-ema-decay=0.995 --reprob=0.5 --smoothing=0.1 --min-lr=1e-8 --warmup-epochs=3 --train-interpolation=bilinear --aa=v0 --model=resnet50 --pretrained --num-classes=100 --opt=adam --weight-decay=1e-4 --log-wandb --dataset-download --experiment "sweep1" --dual --name  "adam lr=1e-4 dual" &





CUDA_VISIBLE_DEVICES=3 python train.py torch/cifar100 --dataset torch/cifar100 -b=128 --img-size=224 --epochs=50 --color-jitter=0 --amp --lr=1e-3 --sched='cosine' --model-ema --model-ema-decay=0.995 --reprob=0.5 --smoothing=0.1 --min-lr=1e-8 --warmup-epochs=3 --train-interpolation=bilinear --aa=v0 --model=resnet50 --pretrained --num-classes=100 --opt=sgd --weight-decay=1e-4 --log-wandb --dataset-download --experiment "sweep1" --metabalance --name  "sgd lr=1e-3 metabalance" &


CUDA_VISIBLE_DEVICES=4 python train.py torch/cifar100 --dataset torch/cifar100 -b=128 --img-size=224 --epochs=50 --color-jitter=0 --amp --lr=1e-4 --sched='cosine' --model-ema --model-ema-decay=0.995 --reprob=0.5 --smoothing=0.1 --min-lr=1e-8 --warmup-epochs=3 --train-interpolation=bilinear --aa=v0 --model=resnet50 --pretrained --num-classes=100 --opt=sgd --weight-decay=1e-4 --log-wandb --dataset-download --experiment "sweep1" --metabalance --name  "sgd lr=1e-4 metabalance" &



CUDA_VISIBLE_DEVICES=6 python train.py torch/cifar100 --dataset torch/cifar100 -b=128 --img-size=224 --epochs=50 --color-jitter=0 --amp --lr=1e-3 --sched='cosine' --model-ema --model-ema-decay=0.995 --reprob=0.5 --smoothing=0.1 --min-lr=1e-8 --warmup-epochs=3 --train-interpolation=bilinear --aa=v0 --model=resnet50 --pretrained --num-classes=100 --opt=adam --weight-decay=1e-4 --log-wandb --dataset-download --experiment "sweep1" --metabalance --name  "adam lr=1e-3 metabalance but fixed 3" &


CUDA_VISIBLE_DEVICES=6 python train.py torch/cifar100 --dataset torch/cifar100 -b=128 --img-size=224 --epochs=50 --color-jitter=0 --amp --lr=1e-4 --sched='cosine' --model-ema --model-ema-decay=0.995 --reprob=0.5 --smoothing=0.1 --min-lr=1e-8 --warmup-epochs=3 --train-interpolation=bilinear --aa=v0 --model=resnet50 --pretrained --num-classes=100 --opt=adam --weight-decay=1e-4 --log-wandb --dataset-download --experiment "sweep1" --metabalance --name  "adam lr=1e-4 metabalance" &






# CUDA_VISIBLE_DEVICES=6 python train.py torch/cifar100 --dataset torch/cifar100 -b=128 --img-size=224 --epochs=50 --color-jitter=0 --amp --lr=1e-4 --sched='cosine' --model-ema --model-ema-decay=0.995 --reprob=0.5 --smoothing=0.1 --min-lr=1e-8 --warmup-epochs=3 --train-interpolation=bilinear --aa=v0 --model=resnet50 --pretrained --num-classes=100 --opt=adam --weight-decay=1e-4 --log-wandb --dataset-download --experiment "sweep1" --adatask --name  "adam lr=1e-4 adatask" &



# sxratch fdist



CUDA_VISIBLE_DEVICES=3,4,5,6 bash distributed_train.sh 4 torch/cifar100 --dataset torch/cifar100 -b=32 --img-size=224 --epochs=50 --color-jitter=0 --amp --lr=1e-3 --sched='cosine' --model-ema --model-ema-decay=0.995 --reprob=0.5 --smoothing=0.1 --min-lr=1e-8 --warmup-epochs=3 --train-interpolation=bilinear --aa=v0 --model=resnet50 --num-classes=100 --opt=adam --weight-decay=1e-4 --log-wandb --dataset-download --experiment "sweep1" --name  "adam lr=1e-3 metabalance dist with adjusted batch" --metabalance



