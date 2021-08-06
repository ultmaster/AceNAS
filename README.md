# AceNAS

This repo is the experiment code of AceNAS, and is not considered as an official release. We are working on integrating AceNAS as a built-in strategy provided in [NNI](https://github.com/microsoft/nni).

## Data Preparation

1. Download our prepared data from [Google Drive](https://drive.google.com/file/d/1Ch_Zwv8NFfY9E12AKcp3X1KRR0yFSExP/view?usp=sharing). The directory should look like this:

```
data
├── checkpoints
│   ├── acenas-m1.pth.tar
│   ├── acenas-m2.pth.tar
│   └── acenas-m3.pth.tar
├── gcn
│   ├── nasbench101_gt_all.pkl
│   ├── nasbench201cifar10_gt_all.pkl
│   ├── nasbench201_gt_all.pkl
│   ├── nasbench201imagenet_gt_all.pkl
│   ├── nds_amoeba_gt_all.pkl
│   ├── nds_amoebaim_gt_all.pkl
│   ├── nds_dartsfixwd_gt_all.pkl
│   ├── nds_darts_gt_all.pkl
│   ├── nds_dartsim_gt_all.pkl
│   ├── nds_enasfixwd_gt_all.pkl
│   ├── nds_enas_gt_all.pkl
│   ├── nds_enasim_gt_all.pkl
│   ├── nds_nasnet_gt_all.pkl
│   ├── nds_nasnetim_gt_all.pkl
│   ├── nds_pnasfixwd_gt_all.pkl
│   ├── nds_pnas_gt_all.pkl
│   ├── nds_pnasim_gt_all.pkl
│   ├── nds_supernet_evaluate_all_test1_amoeba.json
│   ├── nds_supernet_evaluate_all_test1_dartsfixwd.json
│   ├── nds_supernet_evaluate_all_test1_darts.json
│   ├── nds_supernet_evaluate_all_test1_enasfixwd.json
│   ├── nds_supernet_evaluate_all_test1_enas.json
│   ├── nds_supernet_evaluate_all_test1_nasnet.json
│   ├── nds_supernet_evaluate_all_test1_pnasfixwd.json
│   ├── nds_supernet_evaluate_all_test1_pnas.json
│   ├── supernet_evaluate_all_test1_nasbench101.json
│   ├── supernet_evaluate_all_test1_nasbench201cifar10.json
│   ├── supernet_evaluate_all_test1_nasbench201imagenet.json
│   └── supernet_evaluate_all_test1_nasbench201.json
├── nb201
│   ├── split-cifar100.txt
│   ├── split-cifar10-valid.txt
│   └── split-imagenet-16-120.txt
├── proxyless
│   ├── imagenet
│   │   ├── augment_files.txt
│   │   ├── test_files.txt
│   │   ├── train_files.txt
│   │   └── val_files.txt
│   ├── proxyless-84ms-train.csv
│   ├── proxyless-ws-results.csv
│   └── tunas-proxylessnas-search.csv
└── tunas
    ├── imagenet_valid_split_filenames.txt
    ├── random_architectures.csv
    └── searched_architectures.csv
```

2. (Required for benchmark experiments) Download CIFAR-10, CIFAR-100, ImageNet-16-120 dataset and also put them under `data`.

```
data
├── cifar10
│   └── cifar-10-batches-py
│       ├── batches.meta
│       ├── data_batch_1
│       ├── data_batch_2
│       ├── data_batch_3
│       ├── data_batch_4
│       ├── data_batch_5
│       ├── readme.html
│       └── test_batch
├── cifar100
│   └── cifar-100-python
│       ├── meta
│       ├── test
│       └── train
└── imagenet16
    ├── train_data_batch_1
    ├── train_data_batch_10
    ├── train_data_batch_2
    ├── train_data_batch_3
    ├── train_data_batch_4
    ├── train_data_batch_5
    ├── train_data_batch_6
    ├── train_data_batch_7
    ├── train_data_batch_8
    ├── train_data_batch_9
    └── val_data
```

3. (Required for ImageNet experiments) Prepare ImageNet. You can put it anywhere.

4. (Optional) Copy `tunas` (https://github.com/google-research/google-research/tree/master/tunas) to a folder named `tunas`.

## Evaluate pre-trained models.

We provide 3 checkpoints obtained from 3 different runs in `data/checkpoints`. Please evaluate them via the following command.

```bash
python -m tools.standalone.imagenet_eval acenas-m1 /path/to/your/imagenet
python -m tools.standalone.imagenet_eval acenas-m2 /path/to/your/imagenet
python -m tools.standalone.imagenet_eval acenas-m3 /path/to/your/imagenet
```

## Train supernet

```bash
python -m tools.supernet.nasbench101 experiments/supernet/nasbench101.yml
python -m tools.supernet.nasbench201 experiments/supernet/nasbench201.yml
python -m tools.supernet.nds experiments/supernet/darts.yml
python -m tools.supernet.proxylessnas experiments/supernet/proxylessnas.yml
```

Please refer to `experiments/supernet` folder for more configurations.

## Benchmark experiments

We've already provided weight-sharing results from supernet so that you do not have to train you own. The provided files can be found in `json` files located under `data/gcn`.

```bash
# pretrain
python -m gcn.benchmarks.pretrain data/gcn/supernet_evaluate_all_test1_${SEARCHSPACE}.json data/gcn/${SEARCHSPACE}_gt_all.pkl --metric_keys top1 flops params
# finetune
python -m gcn.benchmarks.train --use_train_samples --budget {budget} --test_dataset data/gcn/${SEARCHSPACE}_gt_all.pkl --iteration 5 \
    --loss lambdarank --gnn_type gcn --early_stop_patience 50 --learning_rate 0.005 --opt_type adam --wd 5e-4 --epochs 300 --bs 20 \
    --resume /path/to/previous/output.pt
```

### Running baselines

BRP-NAS:

```bash
# pretrain
python -m gcn.benchmarks.pretrain data/gcn/supernet_evaluate_all_test1_${SEARCHSPACE}.json data/gcn/${SEARCHSPACE}_gt_all.pkl --metric_keys flops
# finetune
python -m gcn.benchmarks.train --use_train_samples --budget ${BUDGET} --test_dataset data/gcn/${SEARCHSPACE}_gt_all.pkl --iteration 5 \
    --loss brp --gnn_type brp --early_stop_patience 35 --learning_rate 0.00035 \
    --opt_type adamw --wd 5e-4 --epochs 250 --bs 64 --resume /path/to/previous/output.pt
```

Vanilla:

```bash
python -m gcn.benchmarks.train --use_train_samples --budget ${BUDGET} --test_dataset data/gcn/${SEARCHSPACE}_gt_all.pkl --iteration 1 \
    --loss mse --gnn_type vanilla --n_hidden 144 --learning_rate 2e-4 --opt_type adam --wd 1e-3 --epochs 300 --bs 10
```

## ProxylessNAS search space

### Train GCN

```bash
python -m gcn.proxyless.pretrain --metric_keys ws_accuracy simulated_pixel1_time_ms flops params
python -m gcn.proxyless.train --loss lambdarank --early_stop_patience 50 --learning_rate 0.002 --opt_type adam --wd 5e-4 --epochs 300 --bs 20 \
    --resume /path/to/previous/output.pth
```

### Train final model

Validation set:

```bash
python -m torch.distributed.launch --nproc_per_node=16 \
    --use_env --module \
    tools.standalone.imagenet_train \
    --output "$OUTPUT_DIR" "$ARCH" "$IMAGENET_DIR" \
    -b 256 --lr 2.64 --warmup-lr 0.1 \
    --warmup-epochs 5 --epochs 90 --sched cosine --num-classes 1000 \
    --opt rmsproptf --opt-eps 1. --weight-decay 4e-5 -j 8 --dist-bn reduce \
    --bn-momentum 0.01 --bn-eps 0.001 --drop 0. --no-held-out-val
```

Test set:

```bash
python -m torch.distributed.launch --nproc_per_node=16 \
    --use_env --module \
    tools.standalone.imagenet_train \
    --output "$OUTPUT_DIR" "$ARCH" "$IMAGENET_DIR" \
    -b 256 --lr 2.64 --warmup-lr 0.1 \
    --warmup-epochs 9 --epochs 360 --sched cosine --num-classes 1000 \
    --opt rmsproptf --opt-eps 1. --weight-decay 4e-5 -j 8 --dist-bn reduce \
    --bn-momentum 0.01 --bn-eps 0.001 --drop 0.15
```
