#!/bin/bash

DIR="./retrained_models"
RUNS=1

# ZINC
python main.py "${DIR}/zinc/gatv2" gatv2 zinc --hidden 104 --final-runs ${RUNS} --hparams "{'lr': 0.004492024637681755, 'batch_size': 128, 'wd': 0.00018406568206249198}"
python main.py "${DIR}/zinc/egc_s" egc zinc --hidden 168 --egc-num-heads 8 --egc-num-bases 4 --aggrs symadd --final-runs ${RUNS} --hparams "{'lr': 0.00278434576243951, 'batch_size': 64, 'wd': 0.00015614444389379077}"
python main.py "${DIR}/zinc/egc_m" egc zinc --hidden 124 --egc-num-heads 4 --egc-num-bases 4 --aggrs add,std,max --final-runs ${RUNS} --hparams "{'lr': 0.0019099809690277627, 'batch_size': 64, 'wd': 0.00020407622034162426}"

# CIFAR
python main.py "${DIR}/cifar/gatv2" gatv2 cifar --hidden 104 --final-runs ${RUNS} --hparams "{'lr': 0.001563799299082841, 'batch_size': 32, 'wd': 0.0003861817258519834, 'dropout': 0.0901933116435249}"
python main.py "${DIR}/cifar/egc_s" egc cifar --hidden 168 --egc-num-heads 8 --egc-num-bases 4 --aggrs symadd --final-runs ${RUNS} --hparams "{'lr': 0.0012354800908953303, 'batch_size': 32, 'wd': 0.000453476392621599, 'dropout': 0.13094687106367725}"
python main.py "${DIR}/cifar/egc_m" egc cifar --hidden 128 --egc-num-heads 4 --egc-num-bases 4 --aggrs symadd,std,max --final-runs ${RUNS} --hparams "{'lr': 0.0009263869626947979, 'batch_size': 32, 'wd': 0.0007592290244995363, 'dropout': 0.08118925150158363}"

# HIV
python main.py "${DIR}/hiv/gcn" gcn hiv --hidden 240 --final-runs ${RUNS} --hparams "{'lr': 0.00031622776601683794, 'batch_size': 32, 'wd': 0.0001, 'dropout': 0.2}"
python main.py "${DIR}/hiv/gat" gat hiv --hidden 240 --final-runs ${RUNS} --hparams "{'lr': 0.00031622776601683794, 'batch_size': 32, 'wd': 0.0001, 'dropout': 0.0}"
python main.py "${DIR}/hiv/gatv2" gatv2 hiv --hidden 184 --final-runs ${RUNS} --hparams "{'lr': 0.00031622776601683794, 'batch_size': 64, 'wd': 0.0001, 'dropout': 0.0}"
python main.py "${DIR}/hiv/gin" gin hiv --hidden 240 --final-runs ${RUNS} --hparams "{'lr': 0.00031622776601683794, 'batch_size': 32, 'wd': 0.0001, 'dropout': 0.2}"
python main.py "${DIR}/hiv/sage" sage hiv --hidden 180 --final-runs ${RUNS} --hparams "{'lr': 0.00031622776601683794, 'batch_size': 64, 'wd': 0.001, 'dropout': 0.2}"
python main.py "${DIR}/hiv/mpnn-max" mpnn-max hiv --hidden 180 --final-runs ${RUNS} --hparams "{'lr': 0.00031622776601683794, 'batch_size': 64, 'wd': 0.0001, 'dropout': 0.0}"
python main.py "${DIR}/hiv/mpnn-sum" mpnn-sum hiv --hidden 180 --final-runs ${RUNS} --hparams "{'lr': 0.0001, 'batch_size': 32, 'wd': 0.001, 'dropout': 0.2}"
python main.py "${DIR}/hiv/egc_s" egc hiv --hidden 296 --egc-num-heads 8 --egc-num-bases 4 --aggrs symadd --final-runs ${RUNS} --hparams "{'lr': 0.0001, 'batch_size': 32, 'wd': 0.0001, 'dropout': 0.2}"
python main.py "${DIR}/hiv/egc_m" egc hiv --hidden 224 --egc-num-heads 4 --egc-num-bases 4 --aggrs add,mean,max --final-runs ${RUNS} --hparams "{'lr': 0.0001, 'batch_size': 32, 'wd': 0.001, 'dropout': 0.2}"

# Arxiv
python main.py "${DIR}/arxiv/gcn" gcn arxiv --hidden 156 --final-runs ${RUNS} --hparams "{'lr': 0.0023853323044733007, 'wd': 0.0001, 'dropout': 0.2}"
python main.py "${DIR}/arxiv/gat" gat arxiv --hidden 152 --final-runs ${RUNS} --hparams "{'lr': 0.0087876393444041, 'wd': 0.0001, 'dropout': 0.2}"
python main.py "${DIR}/arxiv/gatv2" gatv2 arxiv --hidden 112 --final-runs ${RUNS} --hparams "{'lr': 0.0087876393444041, 'wd': 0.001, 'dropout': 0.2}"
python main.py "${DIR}/arxiv/gin" gin arxiv --hidden 156 --final-runs ${RUNS} --hparams "{'lr': 0.0087876393444041, 'wd': 0.0001, 'dropout': 0.2}"
python main.py "${DIR}/arxiv/sage" sage arxiv --hidden 115 --final-runs ${RUNS} --hparams "{'lr': 0.0023853323044733007, 'wd': 0.001, 'dropout': 0.2}"
python main.py "${DIR}/arxiv/mpnn-max" mpnn-max arxiv --hidden 116 --final-runs ${RUNS} --hparams "{'lr': 0.001, 'wd': 0.001, 'dropout': 0.2}"
python main.py "${DIR}/arxiv/mpnn-sum" mpnn-sum arxiv --hidden 116 --final-runs ${RUNS} --hparams "{'lr': 0.03237394014347626, 'wd': 0.0001, 'dropout': 0.2}"
python main.py "${DIR}/arxiv/pna" pna arxiv --hidden 76 --final-runs ${RUNS} --hparams "{'lr': 0.0036840314986403863, 'wd': 0.001, 'dropout': 0.2}"
python main.py "${DIR}/arxiv/egc_s" egc arxiv --hidden 184 --egc-num-heads 8 --egc-num-bases 4 --aggrs symadd --final-runs ${RUNS} --hparams "{'lr': 0.005689810202763908, 'wd': 0.001, 'dropout': 0.2}"
python main.py "${DIR}/arxiv/egc_m" egc arxiv --hidden 136 --egc-num-heads 4 --egc-num-bases 4 --aggrs symadd,max,mean --final-runs ${RUNS} --hparams "{'lr': 0.0036840314986403863, 'wd': 0.001, 'dropout': 0.2}"

# Code2
python main.py "${DIR}/code2/gcn" gcn code --hidden 304 --final-runs ${RUNS} --hparams "{'lr': 0.001584893192461114}"
python main.py "${DIR}/code2/gat" gat code --hidden 304 --final-runs ${RUNS} --hparams "{'lr': 0.00025118864315095795}"
python main.py "${DIR}/code2/gatv2" gatv2 code --hidden 296 --final-runs ${RUNS} --hparams "{'lr': 0.00025118864315095795}"
python main.py "${DIR}/code2/gin" gin code --hidden 304 --final-runs ${RUNS} --hparams "{'lr': 0.001584893192461114}"
python main.py "${DIR}/code2/sage" sage code --hidden 293 --final-runs ${RUNS} --hparams "{'lr': 0.000630957344480193}"
python main.py "${DIR}/code2/mpnn-max" mpnn-max code --hidden 292 --final-runs ${RUNS} --hparams "{'lr': 0.000630957344480193}"
python main.py "${DIR}/code2/mpnn-sum" mpnn-sum code --hidden 292 --final-runs ${RUNS} --hparams "{'lr': 0.00025118864315095795}"
python main.py "${DIR}/code2/pna" pna code --hidden 272 --final-runs ${RUNS} --hparams "{'lr': 0.00063096}"
python main.py "${DIR}/code2/egc_s" egc code --hidden 304 --egc-num-heads 8 --egc-num-bases 8 --aggrs symadd --final-runs ${RUNS} --hparams "{'lr': 0.000630957344480193}"
python main.py "${DIR}/code2/egc_m" egc code --hidden 300 --egc-num-heads 4 --egc-num-bases 4 --aggrs symadd,min,max --final-runs ${RUNS} --hparams "{'lr': 0.001584893192461114}"


# Also included: OGB MAG (for both homogeneous and heterogeneous graphs)
python main.py "${DIR}/mag/mean" egc mag --hidden 352 --final-runs ${RUNS} --egc-num-heads 8 --egc-num-bases 4 --aggrs mean --hparams "{'lr': 0.005, 'wd': 1e-05, 'dropout': 0.3}"
python main.py "${DIR}/mag/symnorm" egc mag --hidden 352 --final-runs ${RUNS} --egc-num-heads 8 --egc-num-bases 4 --aggrs symnorm --hparams "{'lr': 0.01, 'wd': 1e-05, 'dropout': 0.3}"
python main.py "${DIR}/rmag" egc rmag --hidden 64 --final-runs ${RUNS} --egc-num-heads 4 --egc-num-bases 4 --hparams "{'lr': 0.01, 'wd': 0.001, 'dropout': 0.7}"
