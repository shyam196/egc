# Hyperparameters and Experiment Launching

The experiments should be reasonably deterministic given the same set of hyperparameters. However, this has not been tested completely -- although random seeding is done, there are other sources of non-determinism such as CuDNN and scatter operations on the GPU. Regardless, we supply the hyperparameters used in our experiments.

## Launching Runs

You will want to invoke `main.py`. There are 3 positional arguments:

1. Experiment output directory
2. Model
3. Dataset

Optional arguments you will be interested in are:

1. `--hparams` - invokes training of final models using the dictionary of hyperparameters supplied on the command line; the string representation provided is `eval`-ed into the dictionary.
2. `--final-runs` - controls how many final training runs are done; will default to the per-experiment default.
3. `--hidden` - sets the hidden dimension

For EGC experiments, you will want to use the `--egc-num-bases` and `--egc-num-heads` options. You will also want to use the `aggrs` option, which enables setting the choice of aggregators. These are passed by comma separated list; choices include symadd, add, mean, min, max, std and var.

Inspecting the shell scripts in the root of the repo will be useful. If you don't supply a set of hyperparameters, then a new search will be launched.

## Infrastructure Used

We ran our experiments on Nvidia GPUs.
We primarily ran our trials on 1080Tis/2080Tis, with additional experiments being run on V100s and an RTX8000.
Each machine had an Intel Xeon CPU (varying models).

## Hyperparameters

### Zinc

Tune over:

1. Learning rate [0.0001, 0.01]
2. Batch size (64, 128)
3. Weight decay [0.0001, 0.001]

Random Search (50 samples) + hyperband pruning

| Model | Hidden | Bases | Heads | Aggregators |                           Hparams                            |
| :---: | :----: | :---: | :---: | :---------: | :----------------------------------------------------------: |
| GATv2 |  104   |       |       |             | {'lr': 0.004492024637681755, 'batch_size': 128, 'wd': 0.00018406568206249198} |
| EGC-S |  168   |   4   |   8   |   symadd    | {'lr': 0.00278434576243951, 'batch_size': 64, 'wd': 0.00015614444389379077} |
| EGC-M |  124   |   4   |   4   | add,std,max | {'lr': 0.0019099809690277627, 'batch_size': 64, 'wd': 0.00020407622034162426} |

### CIFAR

Tune over:

1. Learning rate [0.0001, 0.01]
2. Batch size (32, 64)
3. Weight decay [0.0001, 0.001]
4. Dropout [0.0, 0.5]

Random search (50 samples) + hyperband pruning

| Model | Hidden | Bases | Heads | Aggregators    | Hparams                                                      |
| ----- | ------ | ----- | ----- | -------------- | ------------------------------------------------------------ |
| GATv2 | 104    |       |       |                | {'lr': 0.001563799299082841, 'batch_size': 32, 'wd': 0.0003861817258519834, 'dropout': 0.0901933116435249} |
| EGC-S | 168    | 4     | 8     | symadd         | {'lr': 0.0012354800908953303, 'batch_size': 32, 'wd': 0.000453476392621599, 'dropout': 0.13094687106367725} |
| EGC-M | 128    | 4     | 4     | symadd,std,max | {'lr': 0.0009263869626947979, 'batch_size': 32, 'wd': 0.0007592290244995363, 'dropout': 0.08118925150158363} |

### MolHIV

Tune over:

1. Learning rate [0.0001, 0.01]
2. Batch size (32, 64)
3. Weight decay [0.0001, 0.001]
4. Dropout [0.0, 0.2]

Grid search (5 LR, 2 BS, 2 weight decay, 2 dropout) + hyperband to prune trials

Dataset version was 1.2.3

| Model    | Hidden | Bases | Heads | Aggregators  | Hparams                                                      |
| -------- | ------ | ----- | ----- | ------------ | ------------------------------------------------------------ |
| GCN      | 240    |       |       |              | {'lr': 0.00031622776601683794, 'batch_size': 32, 'wd': 0.0001, 'dropout': 0.2} |
| GAT      | 240    |       |       |              | {'lr': 0.00031622776601683794, 'batch_size': 32, 'wd': 0.0001, 'dropout': 0.0} |
| GATv2    | 184    |       |       |              | {'lr': 0.00031622776601683794, 'batch_size': 64, 'wd': 0.0001, 'dropout': 0.0} |
| GIN-eps  | 240    |       |       |              | {'lr': 0.00031622776601683794, 'batch_size': 32, 'wd': 0.0001, 'dropout': 0.2} |
| SAGE     | 180    |       |       |              | {'lr': 0.00031622776601683794, 'batch_size': 64, 'wd': 0.001, 'dropout': 0.2} |
| MPNN-Sum | 180    |       |       |              | {'lr': 0.0001, 'batch_size': 32, 'wd': 0.001, 'dropout': 0.2} |
| MPNN-Max | 180    |       |       |              | {'lr': 0.00031622776601683794, 'batch_size': 64, 'wd': 0.0001, 'dropout': 0.0} |
| EGC-S    | 296    | 4     | 8     | symadd       | {'lr': 0.0001, 'batch_size': 32, 'wd': 0.0001, 'dropout': 0.2} |
| EGC-M    | 224    | 4     | 4     | add,mean,max | {'lr': 0.0001, 'batch_size': 32, 'wd': 0.001, 'dropout': 0.2} |

PNA result taken from Corso et al.

### Arxiv

Tune over:

1. Learning rate [0.001, 0.05]
2. Weight decay [0.0001, 0.001]
3. Dropout [0.0, 0.2]

Grid search (5 LR, 2 weight decay, 2 dropout). *No pruning with hyperband*

Dataset version was 1.2.3

| Model    | Hidden | Bases | Heads | Aggregators     | Hparams                                                     |
| -------- | ------ | ----- | ----- | --------------- | ----------------------------------------------------------- |
| GCN      | 156    |       |       |                 | {'lr': 0.0023853323044733007, 'wd': 0.0001, 'dropout': 0.2} |
| GAT      | 152    |       |       |                 | {'lr': 0.0087876393444041, 'wd': 0.0001, 'dropout': 0.2}    |
| GATv2    | 112    |       |       |                 | {'lr': 0.0087876393444041, 'wd': 0.001, 'dropout': 0.2}     |
| GIN-eps  | 156    |       |       |                 | {'lr': 0.0087876393444041, 'wd': 0.0001, 'dropout': 0.2}    |
| SAGE     | 115    |       |       |                 | {'lr': 0.0023853323044733007, 'wd': 0.001, 'dropout': 0.2}  |
| MPNN-Sum | 116    |       |       |                 | {'lr': 0.03237394014347626, 'wd': 0.0001, 'dropout': 0.2}   |
| MPNN-Max | 116    |       |       |                 | {'lr': 0.001, 'wd': 0.001, 'dropout': 0.2}                  |
| PNA      | 76     |       |       |                 | {'lr': 0.0036840314986403863, 'wd': 0.001, 'dropout': 0.2}  |
| EGC-S    | 184    | 4     | 8     | symadd          | {'lr': 0.005689810202763908, 'wd': 0.001, 'dropout': 0.2}   |
| EGC-M    | 136    | 4     | 4     | symadd,max,mean | {'lr': 0.0036840314986403863, 'wd': 0.001, 'dropout': 0.2}  |

### Code-V2

A new version of the code dataset was released in OGB version 1.2.5 after label leakage was discovered in the previous version. We re-ran our experiments on this new version.

Tune over:

1. Learning rate [0.0001, 0.01]

Grid search over 6 learning rates.

| Model    | Hidden | Bases | Heads | Aggregators    | Hparams                        |
| -------- | ------ | ----- | ----- | -------------- | ------------------------------ |
| GCN      | 304    |       |       |                | {'lr': 0.001584893192461114}   |
| GAT      | 304    |       |       |                | {'lr': 0.00025118864315095795} |
| GATv2    | 296    |       |       |                | {'lr': 0.00025118864315095795} |
| GIN-eps  | 304    |       |       |                | {'lr': 0.001584893192461114}   |
| SAGE     | 393    |       |       |                | {'lr': 0.000630957344480193}   |
| MPNN-Sum | 292    |       |       |                | {'lr': 0.00025118864315095795} |
| MPNN-Max | 292    |       |       |                | {'lr': 0.000630957344480193}   |
| PNA      | 272    |       |       |                | {'lr': 0.00063096}             |
| EGC-S    | 304    | 8     | 8     | symadd         | {'lr': 0.000630957344480193}   |
| EGC-M    | 300    | 4     | 4     | symadd,min,max | {'lr': 0.001584893192461114}   |

### OGB-Mag

#### Heterogeneous

Hidden: 64
Heads: 4
Bases: 4
Learning Rate: 0.01
Weight Decay: 0.001
Dropout: 0.7

#### Homogeneous

Hidden size: 352

With symmetric normalization aggregator:
- Heads: 8
- Bases: 4
- Learning Rate: 0.01
- Weight Decay: 1e-5
- Dropout: 0.3

With mean aggregator:
- Heads: 8
- Bases: 4
- Learning Rate: 0.005
- Weight Decay: 1e-5
- Dropout: 0.3

## Ablation Study

**Note**: The hyperparameter tuning approach varies from the approach used above. To reduce variance, we ran each experiment 3 times, and chose the best based on mean *validation* loss across 10 models trained with the same hyperparameters; i.e. we found 3 sets of "best" hyperparameters, and then trained 10 models for each set of "best" hyperparameters. We expect some variance if you re-run the search, but general trends should be preserved: hyperparameter searches use random search on this dataset, and are not seeded; models trained during the search phase are also not seeded. Only the final runs are seeded. In hindsight this is something that we would aim to avoid for future work.

### Parameter Count (100K)

| Heads | Bases | Hidden | Result | Std   | Hyperparameters                                                                |
|-------|-------|--------|--------|-------|--------------------------------------------------------------------------------|
| 1     | 1     | 142    | 0.438  | 0.011 | {'lr': 0.003059716468710177, 'batch_size': 64, 'wd': 0.00024932487158610574}   |
| 1     | 2     | 105    | 0.387  | 0.022 | {'lr': 0.0051846898355138405, 'batch_size': 128, 'wd': 0.00010398947101972665} |
| 1     | 4     | 76     | 0.402  | 0.029 | {'lr': 0.004082569111303931, 'batch_size': 128, 'wd': 0.00021020533452335589}  |
| 1     | 8     | 54     | 0.396  | 0.02  | {'lr': 0.0015950000229818808, 'batch_size': 64, 'wd': 0.00018432566293718268}  |
| 1     | 16    | 39     | 0.397  | 0.017 | {'lr': 0.0016254924999467638, 'batch_size': 64, 'wd': 0.00012255974207700643}  |
|       |       |        |        |       |                                                                                |
| 2     | 1     | 186    | 0.425  | 0.026 | {'lr': 0.004595766114338344, 'batch_size': 128, 'wd': 0.0003406841019850422}   |
| 2     | 2     | 140    | 0.389  | 0.022 | {'lr': 0.006758330977425682, 'batch_size': 128, 'wd': 0.00011116155956381182}  |
| 2     | 4     | 104    | 0.399  | 0.019 | {'lr': 0.0012725522058221876, 'batch_size': 64, 'wd': 0.0003932087276656274}   |
| 2     | 8     | 74     | 0.391  | 0.024 | {'lr': 0.0022763496559145965, 'batch_size': 64, 'wd': 0.00030972140779290556}  |
| 2     | 16    | 52     | 0.419  | 0.011 | {'lr': 0.0009133748989945809, 'batch_size': 64, 'wd': 0.0004191994746174878}   |
|       |       |        |        |       |                                                                                |
| 4     | 1     | 232    | 0.435  | 0.008 | {'lr': 0.0012809553738776893, 'batch_size': 64, 'wd': 0.00023599032344650205}  |
| 4     | 2     | 180    | 0.378  | 0.014 | {'lr': 0.0023503165020786554, 'batch_size': 64, 'wd': 0.00021326255261347772}  |
| 4     | 4     | 136    | 0.405  | 0.019 | {'lr': 0.002280874077442256, 'batch_size': 128, 'wd': 0.00016983733932965093}  |
| 4     | 8     | 100    | 0.397  | 0.023 | {'lr': 0.002157056978525518, 'batch_size': 64, 'wd': 0.0006505102634046523}    |
| 4     | 16    | 68     | 0.4    | 0.028 | {'lr': 0.007475759228858606, 'batch_size': 128, 'wd': 0.00018261215555152014}  |
|       |       |        |        |       |                                                                                |
| 8     | 1     | 272    | 0.408  | 0.014 | {'lr': 0.003237561847876019, 'batch_size': 64, 'wd': 0.0004029760918683308}    |
| 8     | 2     | 216    | 0.388  | 0.022 | {'lr': 0.00565064174990118, 'batch_size': 64, 'wd': 0.00011144780738870096}    |
| 8     | 4     | 168    | 0.364  | 0.02  | {'lr': 0.00278434576243951, 'batch_size': 64, 'wd': 0.00015614444389379077}    |
| 8     | 8     | 120    | 0.396  | 0.029 | {'lr': 0.001406514441546532, 'batch_size': 64, 'wd': 0.00029544213504303457}   |
| 8     | 16    | 80     | 0.391  | 0.022 | {'lr': 0.0036797253125154775, 'batch_size': 128, 'wd': 0.00027104079055315436} |
|       |       |        |        |       |                                                                                |
| 16    | 1     | 288    | 0.391  | 0.015 | {'lr': 0.0061087185167932115, 'batch_size': 128, 'wd': 0.0002273184198789297}  |
| 16    | 2     | 224    | 0.378  | 0.014 | {'lr': 0.00428840518208411, 'batch_size': 64, 'wd': 0.0001061508219486589}     |
| 16    | 4     | 176    | 0.393  | 0.015 | {'lr': 0.002931923031986728, 'batch_size': 128, 'wd': 0.00013034058509380351}  |
| 16    | 8     | 112    | 0.369  | 0.023 | {'lr': 0.003643084029023136, 'batch_size': 128, 'wd': 0.00014767545119931004}  |
| 16    | 16    | 64     | 0.383  | 0.027 | {'lr': 0.0022159422474374592, 'batch_size': 64, 'wd': 0.0001148037568072897}   |

### Hidden Dimension (128)

| Heads | Bases | Result | Std   | Hyperparameters                                                                |
|-------|-------|--------|-------|--------------------------------------------------------------------------------|
| 1     | 1     | 0.457  | 0.026 | {'lr': 0.0009905628560898867, 'batch_size': 64, 'wd': 0.0003791379814624699}   |
| 1     | 2     | 0.402  | 0.029 | {'lr': 0.0033203547862751016, 'batch_size': 128, 'wd': 0.000129368683798121}   |
| 1     | 4     | 0.372  | 0.013 | {'lr': 0.0028185855054290348, 'batch_size': 64, 'wd': 0.000110241337219995}    |
| 1     | 8     | 0.395  | 0.015 | {'lr': 0.0020643416889042335, 'batch_size': 64, 'wd': 0.00024402240952518204}  |
| 1     | 16    | 0.391  | 0.027 | {'lr': 0.0004963552869961618, 'batch_size': 64, 'wd': 0.0003160754920947726}   |
| 2     | 1     | 0.436  | 0.013 | {'lr': 0.0024391466860373976, 'batch_size': 128, 'wd': 0.0003764051687247347}  |
| 2     | 2     | 0.429  | 0.009 | {'lr': 0.0016602428864439917, 'batch_size': 128, 'wd': 0.0002168869706564395}  |
| 2     | 4     | 0.413  | 0.015 | {'lr': 0.0009803643724230493, 'batch_size': 64, 'wd': 0.00011250422859596219}  |
| 2     | 8     | 0.361  | 0.012 | {'lr': 0.0033278618751334138, 'batch_size': 64, 'wd': 0.00010366478664849056}  |
| 2     | 16    | 0.389  | 0.025 | {'lr': 0.002082475573321234, 'batch_size': 128, 'wd': 0.00031036620463648827}  |
| 4     | 1     | 0.431  | 0.009 | {'lr': 0.004700535543388525, 'batch_size': 128, 'wd': 0.0001002933008418741}   |
| 4     | 2     | 0.42   | 0.015 | {'lr': 0.003702073953063512, 'batch_size': 64, 'wd': 0.0006869206027948232}    |
| 4     | 4     | 0.395  | 0.017 | {'lr': 0.003148181818571187, 'batch_size': 128, 'wd': 0.0006299893259191312}   |
| 4     | 8     | 0.388  | 0.016 | {'lr': 0.003723003115072577, 'batch_size': 64, 'wd': 0.000187218003891752}     |
| 4     | 16    | 0.396  | 0.023 | {'lr': 0.0012547863658416598, 'batch_size': 128, 'wd': 0.00018530057376373087} |
| 8     | 1     | 0.421  | 0.025 | {'lr': 0.008110031010018465, 'batch_size': 64, 'wd': 0.00010421914838623193}   |
| 8     | 2     | 0.388  | 0.018 | {'lr': 0.007920837940362318, 'batch_size': 64, 'wd': 0.000113779504056895}     |
| 8     | 4     | 0.395  | 0.006 | {'lr': 0.008610092880667053, 'batch_size': 128, 'wd': 0.00010134943833468606}  |
| 8     | 8     | 0.383  | 0.009 | {'lr': 0.006047352685362815, 'batch_size': 128, 'wd': 0.00021038377802130008}  |
| 8     | 16    | 0.383  | 0.025 | {'lr': 0.0013731703674031866, 'batch_size': 64, 'wd': 0.0002327872787400411}   |
| 16    | 1     | 0.419  | 0.01  | {'lr': 0.005305930382865731, 'batch_size': 64, 'wd': 0.00043968261225748716}   |
| 16    | 2     | 0.401  | 0.019 | {'lr': 0.00257005942283963, 'batch_size': 64, 'wd': 0.0006895449848152606}     |
| 16    | 4     | 0.395  | 0.018 | {'lr': 0.00213094731168947, 'batch_size': 64, 'wd': 0.000862134262819252}      |
| 16    | 8     | 0.382  | 0.019 | {'lr': 0.004052115476867187, 'batch_size': 64, 'wd': 0.00014627932774578965}   |
| 16    | 16    | 0.359  | 0.02  | {'lr': 0.002873665274991742, 'batch_size': 64, 'wd': 0.00013305870959268287}   |
