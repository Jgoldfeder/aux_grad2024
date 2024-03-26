#!/bin/bash

# CUDA_VISIBLE_DEVICES=3 python train.py torch/cifar100 --dataset torch/cifar100 -b=128 --img-size=224 --epochs=50 --color-jitter=0 --amp --lr=1e-2 --sched='cosine' --model-ema --model-ema-decay=0.995 --reprob=0.5 --smoothing=0.1 --min-lr=1e-8 --warmup-epochs=3 --train-interpolation=bilinear --aa=v0 --model=resnet50 --pretrained --num-classes=100 --opt=sgd --weight-decay=1e-4 --log-wandb --dataset-download --experiment "sweep1" --metabalance --name  "sgd lr=1e-2 metabalance" &



# CUDA_VISIBLE_DEVICES=4 python train.py torch/cifar100 --dataset torch/cifar100 -b=128 --img-size=224 --epochs=50 --color-jitter=0 --amp --lr=1e-2 --sched='cosine' --model-ema --model-ema-decay=0.995 --reprob=0.5 --smoothing=0.1 --min-lr=1e-8 --warmup-epochs=3 --train-interpolation=bilinear --aa=v0 --model=resnet50 --pretrained --num-classes=100 --opt=sgd --weight-decay=1e-4 --log-wandb --dataset-download --experiment "sweep1" --dual --name  "sgd lr=1e-2 dual" &


# CUDA_VISIBLE_DEVICES=5 python train.py torch/cifar100 --dataset torch/cifar100 -b=128 --img-size=224 --epochs=50 --color-jitter=0 --amp --lr=1e-2 --sched='cosine' --model-ema --model-ema-decay=0.995 --reprob=0.5 --smoothing=0.1 --min-lr=1e-8 --warmup-epochs=3 --train-interpolation=bilinear --aa=v0 --model=resnet50 --pretrained --num-classes=100 --opt=sgd --weight-decay=1e-4 --log-wandb --dataset-download --experiment "sweep1"  --name  "sgd lr=1e-2 base" &



## cf100

CUDA_VISIBLE_DEVICES=1 python train.py torch/cifar100 --dataset torch/cifar100 -b=128 --img-size=224 --epochs=50 --color-jitter=0 --amp --lr=1e-2 --sched='cosine' --model-ema --model-ema-decay=0.995 --reprob=0.5 --smoothing=0.1 --min-lr=1e-8 --warmup-epochs=3 --train-interpolation=bilinear --aa=v0 --model=resnet50 --pretrained --num-classes=100 --opt=sgd --weight-decay=1e-4 --log-wandb --dataset-download --experiment "sweep1" --metabalance --name  "sgd lr=1e-2 metabalance"

## cf10

CUDA_VISIBLE_DEVICES=1 python train.py --data-dir torch/cifar10 --dataset torch/cifar10 -b=128 --img-size=224 --epochs=50 --color-jitter=0 --amp --lr=1e-2 --sched='cosine' --model-ema --model-ema-decay=0.995 --reprob=0.5 --smoothing=0.1 --min-lr=1e-8 --warmup-epochs=3 --train-interpolation=bilinear --aa=v0 --model=resnet50 --pretrained --num-classes=10 --opt=sgd --weight-decay=1e-4 --log-wandb --dataset-download --experiment "sweep1" --metabalance --name  "sgd lr=1e-2 metabalance"

## mnist

CUDA_VISIBLE_DEVICES=1 python train.py  --data-dir torch/mnist --dataset torch/mnist -b=128 --img-size=224 --epochs=50 --color-jitter=0 --amp --lr=1e-2 --sched='cosine' --model-ema --model-ema-decay=0.995 --reprob=0.5 --smoothing=0.1 --min-lr=1e-8 --warmup-epochs=3 --train-interpolation=bilinear --aa=v0 --model=resnet50 --pretrained --num-classes=10 --opt=sgd --weight-decay=1e-4 --log-wandb --dataset-download --experiment "sweep1" --metabalance --name  "sgd lr=1e-2 metabalance"


## kmnist

CUDA_VISIBLE_DEVICES=1 python train.py --data-dir torch/kmnist --dataset torch/kmnist -b=128 --img-size=224 --epochs=50 --color-jitter=0 --amp --lr=1e-2 --sched='cosine' --model-ema --model-ema-decay=0.995 --reprob=0.5 --smoothing=0.1 --min-lr=1e-8 --warmup-epochs=3 --train-interpolation=bilinear --aa=v0 --model=resnet50 --pretrained --num-classes=10 --opt=sgd --weight-decay=1e-4 --log-wandb --dataset-download --experiment "sweep1" --metabalance --name  "sgd lr=1e-2 metabalance"

## FashionMNIST

CUDA_VISIBLE_DEVICES=1 python train.py --data-dir torch/FashionMNIST --dataset torch/FashionMNIST -b=128 --img-size=224 --epochs=50 --color-jitter=0 --amp --lr=1e-2 --sched='cosine' --model-ema --model-ema-decay=0.995 --reprob=0.5 --smoothing=0.1 --min-lr=1e-8 --warmup-epochs=3 --train-interpolation=bilinear --aa=v0 --model=resnet50 --pretrained --num-classes=10 --opt=sgd --weight-decay=1e-4 --log-wandb --dataset-download --experiment "sweep1" --metabalance --name  "sgd lr=1e-2 metabalance"


## Places365

CUDA_VISIBLE_DEVICES=1 python train.py --data-dir /data/torch/Places365 --dataset torch/Places365 -b=128 --img-size=224 --epochs=50 --color-jitter=0 --amp --lr=1e-2 --sched='cosine' --model-ema --model-ema-decay=0.995 --reprob=0.5 --smoothing=0.1 --min-lr=1e-8 --warmup-epochs=3 --train-interpolation=bilinear --aa=v0 --model=resnet50 --pretrained --num-classes=365 --opt=sgd --weight-decay=1e-4 --log-wandb --dataset-download --experiment "sweep1" --metabalance --name  "sgd lr=1e-2 metabalance"

## inaturalist
CUDA_VISIBLE_DEVICES=1 python train.py --data-dir /data/torch/inaturalist --dataset torch/inaturalist -b=128 --img-size=224 --epochs=50 --color-jitter=0 --amp --lr=1e-2 --sched='cosine' --model-ema --model-ema-decay=0.995 --reprob=0.5 --smoothing=0.1 --min-lr=1e-8 --warmup-epochs=3 --train-interpolation=bilinear --aa=v0 --model=resnet50 --pretrained --num-classes=365 --opt=sgd --weight-decay=1e-4 --log-wandb --dataset-download --experiment "sweep1" --metabalance --name  "sgd lr=1e-2 metabalance"
