#!/bin/bash



# python train_252.py --model resnext50 --dataset stanford_cars --seed 0 --label combo  --epochs 90 --schedule 31 61 --gamma 0.1 --gpus 2 --batch=64 --exp "resnext50_stanford_cars_combination_seed_20"  --combination True --seed 20


# CUDA_VISIBLE_DEVICES=5 python train.py torch/cifar100 --dataset torch/cifar100 -b=128 --img-size=224 --epochs=90 --color-jitter=0 --amp --lr=0.1 --sched='step' --model-ema --model-ema-decay=0.995 --reprob=0.5 --smoothing=0.1 --train-interpolation=bilinear --aa=v0 --model=resnet50  --num-classes=100 --opt=sgd --weight-decay=1e-4 --log-wandb --dataset-download --experiment "sweep1"  --name  "REAL SCRATCH sgd lr=1e-3 base"  --decay-rate 0.1 --decay-epochs 30 &



# CUDA_VISIBLE_DEVICES=6 python train.py torch/cifar100 --dataset torch/cifar100 -b=128 --img-size=224 --epochs=90 --color-jitter=0 --amp --lr=0.1 --sched='step' --model-ema --model-ema-decay=0.995 --reprob=0.5 --smoothing=0.1 --train-interpolation=bilinear --aa=v0 --model=resnet50  --num-classes=100 --opt=sgd --weight-decay=1e-4 --log-wandb --dataset-download --experiment "sweep1"  --name  "REAL SCRATCH sgd lr=1e-3 metabalance_"  --decay-rate 0.1 --decay-epochs 30 --metabalance &





# CUDA_VISIBLE_DEVICES=6 python train.py --data-dir /data/torch/aircraft --dataset torch/aircraft -b=128 --img-size=224 --epochs=90 --color-jitter=0 --amp --lr=0.1 --sched='step' --model-ema --model-ema-decay=0.995 --reprob=0.5 --smoothing=0.1 --train-interpolation=bilinear --aa=v0 --model=resnet50  --num-classes=100 --opt=sgd --weight-decay=1e-4 --log-wandb --dataset-download --experiment "aircraft"  --name  "SCRATCH sgd 90 epochs metabalance_"  --decay-rate 0.1 --decay-epochs 30 --metabalance &

# CUDA_VISIBLE_DEVICES=5 python train.py --data-dir /data/torch/aircraft --dataset torch/aircraft -b=128 --img-size=224 --epochs=90 --color-jitter=0 --amp --lr=0.1 --sched='step' --model-ema --model-ema-decay=0.995 --reprob=0.5 --smoothing=0.1 --train-interpolation=bilinear --aa=v0 --model=resnet50  --num-classes=100 --opt=sgd --weight-decay=1e-4 --log-wandb --dataset-download --experiment "aircraft"  --name  "SCRATCH sgd 90 epochs base"  --decay-rate 0.1 --decay-epochs 30  &



# CUDA_VISIBLE_DEVICES=4 python train.py --data-dir /data/torch/aircraft --dataset torch/aircraft -b=128 --img-size=224 --epochs=90 --color-jitter=0 --amp --lr=0.1 --sched='step' --model-ema --model-ema-decay=0.995 --reprob=0.5 --smoothing=0.1 --train-interpolation=bilinear --aa=v0 --model=resnet50  --num-classes=100 --opt=sgd --weight-decay=1e-4 --log-wandb --dataset-download --experiment "aircraft"  --name  "pretrained sgd 90 epochs metabalance_"  --decay-rate 0.1 --decay-epochs 30 --metabalance --pretrained &

# CUDA_VISIBLE_DEVICES=0 python train.py --data-dir /data/torch/aircraft --dataset torch/aircraft -b=128 --img-size=224 --epochs=90 --color-jitter=0 --amp --lr=0.1 --sched='step' --model-ema --model-ema-decay=0.995 --reprob=0.5 --smoothing=0.1 --train-interpolation=bilinear --aa=v0 --model=resnet50  --num-classes=100 --opt=sgd --weight-decay=1e-4 --log-wandb --dataset-download --experiment "aircraft"  --name  "pretrained sgd 90 epochs base noaug"  --decay-rate 0.1 --decay-epochs 30 --pretrained --no-aug &




# CUDA_VISIBLE_DEVICES=1 python train.py --data-dir /data/torch/food101 --dataset torch/food101 -b=128 --img-size=224 --epochs=90 --color-jitter=0 --amp --lr=0.1 --sched='step' --model-ema --model-ema-decay=0.995 --reprob=0.5 --smoothing=0.1 --train-interpolation=bilinear --aa=v0 --model=resnet50  --num-classes=102 --opt=sgd --weight-decay=1e-4 --log-wandb --dataset-download --experiment "food101"  --name  "pretrained sgd 90 epochs metabalance_"  --decay-rate 0.1 --decay-epochs 30 --metabalance --pretrained  &

# CUDA_VISIBLE_DEVICES=2 python train.py --data-dir /data/torch/food101 --dataset torch/food101 -b=128 --img-size=224 --epochs=90 --color-jitter=0 --amp --lr=0.1 --sched='step' --model-ema --model-ema-decay=0.995 --reprob=0.5 --smoothing=0.1 --train-interpolation=bilinear --aa=v0 --model=resnet50  --num-classes=102 --opt=sgd --weight-decay=1e-4 --log-wandb --dataset-download --experiment "food101"  --name  "pretrained sgd 90 epochs base"  --decay-rate 0.1 --decay-epochs 30 --pretrained  &



########## HERE


#torch/dogs


CUDA_VISIBLE_DEVICES=6 python train.py --data-dir /data/torch/country211  --dataset torch/country211 -b=128 --img-size=224 --epochs=90 --color-jitter=0 --amp --lr=0.001 --sched='step' --model-ema --model-ema-decay=0.995 --reprob=0.5 --smoothing=0.1 --train-interpolation=bilinear --aa=v0 --model=resnet50  --num-classes=211 --opt=adam --weight-decay=1e-4 --log-wandb --dataset-download --experiment "country211"  --name  "Pretrain adam 90 epochs metabalance_ fc1"  --decay-rate 0.1 --decay-epochs 30 --metabalance --pretrained &

CUDA_VISIBLE_DEVICES=5 python train.py --data-dir /data/torch/country211  --dataset torch/country211 -b=128 --img-size=224 --epochs=90 --color-jitter=0 --amp --lr=0.001 --sched='step' --model-ema --model-ema-decay=0.995 --reprob=0.5 --smoothing=0.1 --train-interpolation=bilinear --aa=v0 --model=resnet50  --num-classes=211 --opt=adam --weight-decay=1e-4 --log-wandb --dataset-download --experiment "country211"  --name  "Pretrain adam 90 epochs base fc1 "  --decay-rate 0.1 --decay-epochs 30  --pretrained &



CUDA_VISIBLE_DEVICES=4 python train.py --data-dir /data/torch/country211  --dataset torch/country211 -b=128 --img-size=224 --epochs=90 --color-jitter=0 --amp --lr=0.001 --sched='step' --model-ema --model-ema-decay=0.995 --reprob=0.5 --smoothing=0.1 --train-interpolation=bilinear --aa=v0 --model=resnet50  --num-classes=211 --opt=adam --weight-decay=1e-4 --log-wandb --dataset-download --experiment "country211"  --name  "SCRATCH adam 90 epochs metabalance_ fc1"  --decay-rate 0.1 --decay-epochs 30 --metabalance &

CUDA_VISIBLE_DEVICES=3 python train.py --data-dir /data/torch/country211  --dataset torch/country211 -b=128 --img-size=224 --epochs=90 --color-jitter=0 --amp --lr=0.001 --sched='step' --model-ema --model-ema-decay=0.995 --reprob=0.5 --smoothing=0.1 --train-interpolation=bilinear --aa=v0 --model=resnet50  --num-classes=211 --opt=adam --weight-decay=1e-4 --log-wandb --dataset-download --experiment "country211"  --name  "SCRATCH adam 90 epochs base fc1"  --decay-rate 0.1 --decay-epochs 30   &




CUDA_VISIBLE_DEVICES=2 python train.py --data-dir /data/torch/caltech256  --dataset torch/caltech256 -b=128 --img-size=224 --epochs=90 --color-jitter=0 --amp --lr=0.1 --sched='step' --model-ema --model-ema-decay=0.995 --reprob=0.5 --smoothing=0.1 --train-interpolation=bilinear --aa=v0 --model=resnet50  --num-classes=257 --opt=sgd --weight-decay=1e-4 --log-wandb --dataset-download --experiment "caltech256"  --name  "SCRATCH sgd 90 epochs metabalance_"  --decay-rate 0.1 --decay-epochs 30 --metabalance &

# CUDA_VISIBLE_DEVICES=1 python train.py --data-dir /data/torch/caltech256  --dataset torch/caltech256 -b=128 --img-size=224 --epochs=90 --color-jitter=0 --amp --lr=0.1 --sched='step' --model-ema --model-ema-decay=0.995 --reprob=0.5 --smoothing=0.1 --train-interpolation=bilinear --aa=v0 --model=resnet50  --num-classes=257 --opt=sgd --weight-decay=1e-4 --log-wandb --dataset-download --experiment "caltech256"  --name  "SCRATCH sgd 90 epochs base"  --decay-rate 0.1 --decay-epochs 30   &






# CUDA_VISIBLE_DEVICES=0 python train.py --data-dir /data/hfds/stanford_cars  --dataset hfds/stanford_cars -b=128 --img-size=224 --epochs=90 --color-jitter=0 --amp --lr=0.1 --sched='step' --model-ema --model-ema-decay=0.995 --reprob=0.5 --smoothing=0.1 --train-interpolation=bilinear --aa=v0 --model=resnet50  --num-classes=196 --opt=sgd --weight-decay=1e-4 --log-wandb --dataset-download --experiment "stanford_cars"  --name  "SCRATCH sgd 90 epochs base simple"  --decay-rate 0.1 --decay-epochs 30  --simpleloader &

# CUDA_VISIBLE_DEVICES=1 python train.py --data-dir /data/hfds/stanford_cars  --dataset hfds/stanford_cars -b=128 --img-size=224 --epochs=90 --color-jitter=0 --amp --lr=0.1 --sched='step' --model-ema --model-ema-decay=0.995 --reprob=0.5 --smoothing=0.1 --train-interpolation=bilinear --aa=v0 --model=resnet50  --num-classes=196 --opt=sgd --weight-decay=1e-4 --log-wandb --dataset-download --experiment "stanford_cars"  --name  "SCRATCH sgd 90 epochs metabalance_ simple"  --decay-rate 0.1 --decay-epochs 30 --metabalance --simpleloader &

# CUDA_VISIBLE_DEVICES=2 python train.py --data-dir /data/hfds/stanford_cars  --dataset hfds/stanford_cars -b=128 --img-size=224 --epochs=90 --color-jitter=0 --amp --lr=0.1 --sched='step' --model-ema --model-ema-decay=0.995 --reprob=0.5 --smoothing=0.1 --train-interpolation=bilinear --aa=v0 --model=resnet50  --num-classes=196 --opt=sgd --weight-decay=1e-4 --log-wandb --dataset-download --experiment "stanford_cars"  --name  "Pretrain sgd 90 epochs base"  --decay-rate 0.1 --decay-epochs 30  --pretrained &

# CUDA_VISIBLE_DEVICES=3 python train.py --data-dir /data/hfds/stanford_cars  --dataset hfds/stanford_cars -b=128 --img-size=224 --epochs=90 --color-jitter=0 --amp --lr=0.1 --sched='step' --model-ema --model-ema-decay=0.995 --reprob=0.5 --smoothing=0.1 --train-interpolation=bilinear --aa=v0 --model=resnet50  --num-classes=196 --opt=sgd --weight-decay=1e-4 --log-wandb --dataset-download --experiment "stanford_cars"  --name  "Pretrain sgd 90 epochs metabalance"  --metabalance --decay-rate 0.1 --decay-epochs 30  --pretrained &







# CUDA_VISIBLE_DEVICES=4 python train.py --data-dir /data/hfds/stanford_cars  --dataset hfds/stanford_cars -b=128 --img-size=224 --epochs=90 --color-jitter=0 --amp --lr=0.001 --sched='step' --model-ema --model-ema-decay=0.995 --reprob=0.5 --smoothing=0.1 --train-interpolation=bilinear --aa=v0 --model=resnet50  --num-classes=196 --opt=adam --weight-decay=1e-4 --log-wandb --dataset-download --experiment "stanford_cars"  --name  "SCRATCH adam 90 epochs base simple"  --decay-rate 0.1 --decay-epochs 30  --simpleloader &



# CUDA_VISIBLE_DEVICES=5 python train.py --data-dir /data/hfds/stanford_cars  --dataset hfds/stanford_cars -b=128 --img-size=224 --epochs=90 --color-jitter=0 --amp --lr=0.001 --sched='step' --model-ema --model-ema-decay=0.995 --reprob=0.5 --smoothing=0.1 --train-interpolation=bilinear --aa=v0 --model=resnet50  --num-classes=196 --opt=adam --weight-decay=1e-4 --log-wandb --dataset-download --experiment "stanford_cars"  --name  "SCRATCH adam 90 epochs metabalance simple"  --decay-rate 0.1 --decay-epochs 30  --simpleloader --metabalance &


CUDA_VISIBLE_DEVICES=4 python train.py --data-dir /data/hfds/stanford_cars  --dataset hfds/stanford_cars -b=128 --img-size=224 --epochs=90 --color-jitter=0 --amp --lr=0.001 --sched='step' --model-ema --model-ema-decay=0.995 --reprob=0.5 --smoothing=0.1 --train-interpolation=bilinear --aa=v0 --model=resnet50  --num-classes=196 --opt=adam --weight-decay=1e-4 --log-wandb --dataset-download --experiment "stanford_cars"  --name  "pretrain adam 90 epochs metabalance fc1   1 9"  --decay-rate 0.1 --decay-epochs 30   --metabalance  --pretrained --weights 1 9 &




CUDA_VISIBLE_DEVICES=3 python train.py --data-dir /data/hfds/stanford_cars  --dataset hfds/stanford_cars -b=128 --img-size=224 --epochs=90 --color-jitter=0 --amp --lr=0.001 --sched='step' --model-ema --model-ema-decay=0.995 --reprob=0.5 --smoothing=0.1 --train-interpolation=bilinear --aa=v0 --model=resnet50  --num-classes=196 --opt=adam --weight-decay=1e-4 --log-wandb --dataset-download --experiment "stanford_cars"  --name  "lvl 5 pretrain adam 90 epochs metabalance fc1 simple"  --decay-rate 0.1 --decay-epochs 30   --metabalance --pretrained --level 5 --simpleloader &

CUDA_VISIBLE_DEVICES=4 python train.py --data-dir /data/hfds/stanford_cars  --dataset hfds/stanford_cars -b=128 --img-size=224 --epochs=90 --color-jitter=0 --amp --lr=0.001 --sched='step' --model-ema --model-ema-decay=0.995 --reprob=0.5 --smoothing=0.1 --train-interpolation=bilinear --aa=v0 --model=resnet50  --num-classes=196 --opt=adam --weight-decay=1e-4 --log-wandb --dataset-download --experiment "stanford_cars"  --name  "lvl 5 pretrain adam 90 epochs base fc1 simple"  --decay-rate 0.1 --decay-epochs 30    --pretrained --level 5 --simpleloader &



CUDA_VISIBLE_DEVICES=6 python train_252.py --model resnet50 --dataset stanford_cars --seed 0 --label combo  --epochs 90 --schedule 31 61 --gamma 0.1 --gpus 0 --batch=64 --exp "resnet50_stanford_cars_combination_seed_10"  --combination True  --seed 10 &

# CUDA_VISIBLE_DEVICES=5 python train_252.py --model resnet50 --dataset stanford_cars --seed 0 --label category  --epochs 90 --schedule 31 61 --gamma 0.1 --gpus 0 --batch=64 --exp "resnet50_stanford_cars_category_seed_10"   --seed 10 &

# CUDA_VISIBLE_DEVICES=4 python train_252.py --model resnet50 --dataset caltech101 --seed 0 --label combo  --epochs 90 --schedule 31 61 --gamma 0.1 --gpus 0 --batch=64 --exp "resnet50_caltech101_combination_seed_10"  --combination True  --seed 10 &

# CUDA_VISIBLE_DEVICES=3 python train_252.py --model resnet50 --dataset caltech101 --seed 0 --label category  --epochs 90 --schedule 31 61 --gamma 0.1 --gpus 0 --batch=64 --exp "resnet50_caltech101_category_seed_10"   --seed 10 &

# CUDA_VISIBLE_DEVICES=1 python train_252_updated.py --model resnet50 --dataset aircraft --seed 0 --label combo  --epochs 90 --schedule 31 61 --gamma 0.1 --gpus 0 --batch=64 --exp "resnet50_aircraft_combination_seed_10"  --combination True  --seed 10 &

# CUDA_LAUNCH_BLOCKING=1  CUDA_VISIBLE_DEVICES=0 python train_252_updated.py --model resnet50 --dataset aircraft --seed 0 --label category  --epochs 90 --schedule 31 61 --gamma 0.1 --gpus 0 --batch=64 --exp "resnet50_aircraft_category_seed_10"   --seed 10 &



CUDA_VISIBLE_DEVICES=1 python train_252_updated.py --model resnext50 --dataset aircraft --seed 0 --label combo  --epochs 90 --schedule 31 61 --gamma 0.1 --gpus 0 --batch=64 --exp "resnext50_aircraft_combination_seed_10"  --combination True --seed 10 &

CUDA_VISIBLE_DEVICES=2 python train_252_updated.py --model resnext50 --dataset aircraft --seed 0 --label category  --epochs 90 --schedule 31 61 --gamma 0.1 --gpus 0 --batch=64 --exp "resnext50_aircraft_category_seed_10"   --seed 10 &

CUDA_VISIBLE_DEVICES=? python train_252.py --model resnet101 --dataset aircraft --seed 0 --label combo  --epochs 90 --schedule 31 61 --gamma 0.1 --gpus 0 --batch=64 --exp "resnet101_aircraft_combination_seed_10"  --combination True --seed 10 &


