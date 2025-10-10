# Error-Driven Uncertainty Aware Training (EUAT)

This repository contains the scripts and data used to evaluate EUAT.


##  Train directory

**This directory contains the scripts and information to deploy the training of models.**

To train the models, you can directly run the `python3 train.py` specifying the correct arguments:
```
  --type: type of training (standard training (AT) or adversarial training (AT)), choices=['std', 'robust']
  --dataset: dataset/model to train, choices=['mnist', 'cifar10', 'cifar10-c', 'binaryCifar10', 'cifar100', 'imageNet', 'svhn']

  --stop: stop condition, choices=['epochs', 'time']
  --stop_val: the value of the stop condition
  --lr: learning rate 
  --momentum:  momentum 
  --batch:  batch size
  --workers: number of workers
  --half_prec: half-precision computation
  --variants: different baselines, default='none', choices=['none','calibration','deup','ensemble']
```


When performing adversarial training (AT), you need to consider the following arguments:
```
  --alg: method to generate the perturbation when performing adversarial training, choices=['pgd', 'fgsm']
  --ratio: percentage of resources for AT (%RAT), value between 0 and 1
  --ratio_adv: percentage of adversarial examples (%AE) when performing AT, , value between 0 and 1
  --epsilon: bound for the pertubation
  --num_iter: number of iterations for PGD
  --alpha: similar to the learning rate for PGD
  --lr_adv: learning rate of AT
  --momentum_adv: momentum of AT
  --batch_adv: the batch size of AT
```

The benchmarks should be in a directory called `../data`.
After training, the models are save in `../models/` directory and the logs files are saved in `logs/`.
If the models exist in `../models/` when start training, then it should exist the corresponding log file in `../logs1/` and the model and the logs will be loaded.
You can easily change these directories.

To parallelize and automatize the deployment and speed up the training, you can also run `python3 run.py`. In this file, you can directly specify all the arguments to run the train.py file, and the script will deploy in a sequential (1 worker) or parallel way (several workers) the training process.
Do not forget to install pytorch, torchensemble, and netcal.


## Acknowledgments

This work was supported by the Fundação para a Ciência e a Tecnologia (Portuguese Foundation for Science and Technology) through the Carnegie Mellon Portugal Program under grant SFRH/BD/151470/2021, and by projects UIDB/50021/2020,  C645008882-00000055.PRR and C628696807-00454142 (Center for Responsible AI), 101189689 (ACHILLES).
This work was developed within the scope of the project no. 62 - "Responsible A", financed by European Funds, namely "Recovery and Resilience Plan - Component 5: Agendas Mobilizadoras para a Inovação Empresarial", included in the NextGenerationEU funding program.

![Alt text](./image.png)
