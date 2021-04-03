#!/bin/bash

DIR="./retrained_models"
RUNS=1

# Constant parameter count
python main.py "${DIR}/param_ablation/h4b4" egc zinc --hidden 136 --egc-num-heads 4 --egc-num-bases 4 --aggrs symadd --final-runs ${RUNS} --hparams "{'lr': 0.002280874077442256, 'batch_size': 128, 'wd': 0.00016983733932965093}"
python main.py "${DIR}/param_ablation/h4b8" egc zinc --hidden 100 --egc-num-heads 4 --egc-num-bases 8 --aggrs symadd --final-runs ${RUNS} --hparams "{'lr': 0.002157056978525518, 'batch_size': 64, 'wd': 0.0006505102634046523}"
python main.py "${DIR}/param_ablation/h4b16" egc zinc --hidden 68 --egc-num-heads 4 --egc-num-bases 16 --aggrs symadd --final-runs ${RUNS} --hparams "{'lr': 0.007475759228858606, 'batch_size': 128, 'wd': 0.00018261215555152014}"
python main.py "${DIR}/param_ablation/h8b4" egc zinc --hidden 168 --egc-num-heads 8 --egc-num-bases 4 --aggrs symadd --final-runs ${RUNS} --hparams "{'lr': 0.00278434576243951, 'batch_size': 64, 'wd': 0.00015614444389379077}"
python main.py "${DIR}/param_ablation/h8b8" egc zinc --hidden 120 --egc-num-heads 8 --egc-num-bases 8 --aggrs symadd --final-runs ${RUNS} --hparams "{'lr': 0.001406514441546532, 'batch_size': 64, 'wd': 0.00029544213504303457}"
python main.py "${DIR}/param_ablation/h8b16" egc zinc --hidden 80 --egc-num-heads 8 --egc-num-bases 16 --aggrs symadd --final-runs ${RUNS} --hparams "{'lr': 0.0036797253125154775, 'batch_size': 128, 'wd': 0.00027104079055315436}"
python main.py "${DIR}/param_ablation/h16b4" egc zinc --hidden 176 --egc-num-heads 16 --egc-num-bases 4 --aggrs symadd --final-runs ${RUNS} --hparams "{'lr': 0.002931923031986728, 'batch_size': 128, 'wd': 0.00013034058509380351}"
python main.py "${DIR}/param_ablation/h16b8" egc zinc --hidden 112 --egc-num-heads 16 --egc-num-bases 8 --aggrs symadd --final-runs ${RUNS} --hparams "{'lr': 0.003643084029023136, 'batch_size': 128, 'wd': 0.00014767545119931004}"
python main.py "${DIR}/param_ablation/h16b16" egc zinc --hidden 64 --egc-num-heads 16 --egc-num-bases 16 --aggrs symadd --final-runs ${RUNS} --hparams "{'lr': 0.0022159422474374592, 'batch_size': 64, 'wd': 0.0001148037568072897}"


# Constant hidden
python main.py "${DIR}/headbase_ablation/h4b4" egc zinc --hidden 128 --egc-num-heads 4 --egc-num-bases 4 --aggrs symadd --final-runs ${RUNS} --hparams "{'lr': 0.003148181818571187, 'batch_size': 128, 'wd': 0.0006299893259191312}"
python main.py "${DIR}/headbase_ablation/h4b8" egc zinc --hidden 128 --egc-num-heads 4 --egc-num-bases 8 --aggrs symadd --final-runs ${RUNS} --hparams "{'lr': 0.003723003115072577, 'batch_size': 64, 'wd': 0.000187218003891752}"
python main.py "${DIR}/headbase_ablation/h4b16" egc zinc --hidden 128 --egc-num-heads 4 --egc-num-bases 16 --aggrs symadd --final-runs ${RUNS} --hparams "{'lr': 0.0012547863658416598, 'batch_size': 128, 'wd': 0.00018530057376373087}"
python main.py "${DIR}/headbase_ablation/h8b4" egc zinc --hidden 128 --egc-num-heads 8 --egc-num-bases 4 --aggrs symadd --final-runs ${RUNS} --hparams "{'lr': 0.008610092880667053, 'batch_size': 128, 'wd': 0.00010134943833468606}"
python main.py "${DIR}/headbase_ablation/h8b8" egc zinc --hidden 128 --egc-num-heads 8 --egc-num-bases 8 --aggrs symadd --final-runs ${RUNS} --hparams "{'lr': 0.006047352685362815, 'batch_size': 128, 'wd': 0.00021038377802130008}"
python main.py "${DIR}/headbase_ablation/h8b16" egc zinc --hidden 128 --egc-num-heads 8 --egc-num-bases 16 --aggrs symadd --final-runs ${RUNS} --hparams "{'lr': 0.0013731703674031866, 'batch_size': 64, 'wd': 0.0002327872787400411}"
python main.py "${DIR}/headbase_ablation/h16b4" egc zinc --hidden 128 --egc-num-heads 16 --egc-num-bases 4 --aggrs symadd --final-runs ${RUNS} --hparams "{'lr': 0.00213094731168947, 'batch_size': 64, 'wd': 0.000862134262819252}"
python main.py "${DIR}/headbase_ablation/h16b8" egc zinc --hidden 128 --egc-num-heads 16 --egc-num-bases 8 --aggrs symadd --final-runs ${RUNS} --hparams "{'lr': 0.004052115476867187, 'batch_size': 64, 'wd': 0.00014627932774578965}"
python main.py "${DIR}/headbase_ablation/h16b16" egc zinc --hidden 128 --egc-num-heads 16 --egc-num-bases 16 --aggrs symadd --final-runs ${RUNS} --hparams "{'lr': 0.002873665274991742, 'batch_size': 64, 'wd': 0.00013305870959268287}"
