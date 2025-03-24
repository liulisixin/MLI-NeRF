# MLI-NeRF (Updating)
The official implement of "MLI-NeRF: Multi-Light Intrinsic-Aware Neural Radiance Fields".

Yixiong Yang, Shilin Hu, Haoyu Wu, Ramon Baldrich, Dimitris Samaras, Maria Vanrell

International Conference on 3D Vision 2025

### | [Paper](https://arxiv.org/abs/2411.17235) |

https://github.com/user-attachments/assets/0ae8436f-a2af-4c07-8a96-b504db106844

## Requirements
We use [Neuralangelo](https://github.com/NVlabs/neuralangelo) as the baseline model. Therefore, we recommend following the installation instructions on the Neuralangelo website (without the need to install COLMAP).

Tips:
1. The code requires tiny-cuda-nn, which needs the CUDA version of PyTorch to be consistent with the CUDA version installed on your system.
2. Use pip to install pymcubes==0.1.4.

## Run on the real object dataset
Let's use one example from the real object dataset.

Download dataset from:
https://github.com/iamNCJ/NRHints
Modify the path in CONF_a and CONF_b.

Train and test
```angular2html
CONF_a="NRHints_Pikachu_a"
CONF_b="NRHints_Pikachu_b"
python train.py --logdir=logs --config=projects/NeuralLumen/configs/${CONF_a}.yaml --show_pbar --single_gpu --wandb --wandb_name=angelo
python test.py --config=projects/NeuralLumen/configs/${CONF_a}.yaml --show_pbar --single_gpu --inference_mode unpairlights_train --model.light_visibility.enabled=True --model.render.rand_rays_val=10000
python projects/NeuralLumen/scripts/pseudo_label.py --workdir /ghome/yyang/PycharmProjects/neuralangelo/logs/${CONF_b}/output_unpairlights --setting unpair
python train.py --logdir=logs --config=projects/NeuralLumen/configs/${CONF_b}.yaml --show_pbar --single_gpu --wandb --wandb_name=angelo
python test.py --config=projects/NeuralLumen/configs/${CONF_b}.yaml --show_pbar --single_gpu --inference_mode image_test --anno transforms_test.json
```
Video results
```angular2html
CONF_b="NRHints_Pikachu_b"
python test.py --config=projects/NeuralLumen/configs/${CONF_b}.yaml --show_pbar --single_gpu --inference_mode video_train_0_67
```


## Run on the synthetic dataset
One example from the synthetic dataset. (The datasets can be downloaded from (TBD))
Modify the path in CONF_a and CONF_b.

Train and test
```angular2html
CONF_a="syn_hotdog_a"
CONF_b="syn_hotdog_b"
python train.py --logdir=logs --config=projects/NeuralLumen/configs/${CONF_a}.yaml --show_pbar --single_gpu --wandb --wandb_name=angelo
python test.py --config=projects/NeuralLumen/configs/${CONF_a}.yaml --show_pbar --single_gpu --inference_mode unpairlights_train --model.light_visibility.enabled=True --model.render.rand_rays_val=10000
python projects/NeuralLumen/scripts/pseudo_label.py --workdir /ghome/yyang/PycharmProjects/neuralangelo/logs/${CONF_b}/output_unpairlights --setting unpair
python train.py --logdir=logs --config=projects/NeuralLumen/configs/${CONF_b}.yaml --show_pbar --single_gpu --wandb --wandb_name=angelo
python test.py --config=projects/NeuralLumen/configs/${CONF_b}.yaml --show_pbar --single_gpu --inference_mode image_test --anno transforms_test.json
```

Video results
```angular2html
python test.py --config=projects/NeuralLumen/configs/${CONF_b}.yaml --show_pbar --single_gpu --inference_mode video_train_0_67
```


## Run on the ReNe dataset
The ReNe dataset can be downloaded from [ReNe](https://github.com/eyecan-ai/rene).  
We provide a script, `projects/NeuralLumen/scripts/convert_rene_direct_to_json.py`, to generate the JSON files.  
**Note**: This script requires the ReNe dataset package to be installed beforehand.  

The generated JSON files are also included in `dataset_rene`.


```angular2html
CONF_a="rene_savannah_a"
CONF_b="rene_savannah_b"
python train.py --logdir=logs --config=projects/NeuralLumen/configs/${CONF_a}.yaml --show_pbar --single_gpu --wandb --wandb_name=angelo
python test.py --config=projects/NeuralLumen/configs/${CONF_a}.yaml --show_pbar --single_gpu --inference_mode lights_train --model.light_visibility.enabled=True --model.render.rand_rays_val=10000
python projects/NeuralLumen/scripts/pseudo_label.py --workdir /ghome/yyang/PycharmProjects/neuralangelo/logs/${CONF_a}/output_lights --setting pair
python train.py --logdir=logs --config=projects/NeuralLumen/configs/${CONF_b}.yaml --show_pbar --single_gpu --wandb --wandb_name=angelo
python test.py --config=projects/NeuralLumen/configs/${CONF_b}.yaml --show_pbar --single_gpu --inference_mode image_test --anno test_custom_transforms.json
```
Video output
```angular2html
python test.py --config=projects/NeuralLumen/configs/${CONF_b}.yaml --show_pbar --single_gpu --inference_mode video_train_0_67
```

# Citation
Please cite this repository as follows if you find it helpful for your project :):
```
@inproceedings{yang2024mlinerf,
  title={MLI-NeRF: Multi-Light Intrinsic-Aware Neural Radiance Fields},
  author={Yixiong Yang and Shilin Hu and Haoyu Wu and Ramon Baldrich and Dimitris Samaras and Maria Vanrell},
  booktitle={The International Conference on 3D Vision},
  year={2025}
}
```

## Acknowledgments
Some codes are borrowed from [Neuralangelo](https://github.com/NVlabs/neuralangelo) and [NRHints](https://github.com/iamNCJ/NRHints). Thanks for their great works.

