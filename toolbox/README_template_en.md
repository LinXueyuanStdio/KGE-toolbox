# KGE Toolbox


## Environment

create a conda environment with `pytorch` `cython` and `scikit-learn` :
```shell
conda create --name toolbox_env python=3.7
source activate toolbox_env
conda install --file requirements.txt -c pytorch
```
## How to run

```shell
python train.py --batch_size=512 --name=TryMyModel
```
