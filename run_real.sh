#!/bin/bash
#SBATCH -n 6 # Number of cores
#SBATCH --mem 20096 # 4GB solicitados.
#SBATCH -p dcca40 # or mlow Partition to submit to master low prioriy queue
#SBATCH --gres gpu:1 # Para pedir Pascales MAX 8
#SBATCH -o %x_%u_%j.out # File to which STDOUT will be written
#SBATCH -e %x_%u_%j.err # File to which STDERR will be written

CONF_a="NRHints_Pikachu_a"
CONF_b="NRHints_Pikachu_b"

python train.py --logdir=logs --config=projects/NeuralLumen/configs/${CONF_a}.yaml --show_pbar --single_gpu --wandb --wandb_name=angelo
python test.py --config=projects/NeuralLumen/configs/${CONF_a}.yaml --show_pbar --single_gpu --inference_mode unpairlights_train --model.light_visibility.enabled=True --model.render.rand_rays_val=10000
python projects/NeuralLumen/scripts/pseudo_label.py --workdir ./logs/${CONF_a}/output_unpairlights --setting unpair
python train.py --logdir=logs --config=projects/NeuralLumen/configs/${CONF_b}.yaml --show_pbar --single_gpu --wandb --wandb_name=angelo
python test.py --config=projects/NeuralLumen/configs/${CONF_b}.yaml --show_pbar --single_gpu --inference_mode image_test --anno transforms_test.json

# python test.py --config=projects/NeuralLumen/configs/${CONF_b}.yaml --show_pbar --single_gpu --inference_mode video_train_0_67

