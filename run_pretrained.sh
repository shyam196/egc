#!/bin/bash

DIR="./pretrained"

# ZINC
python main.py "${DIR}" gatv2 zinc --pretrained --hidden 104
python main.py "${DIR}" egc zinc --pretrained --hidden 168 --egc-num-heads 8 --egc-num-bases 4 --aggrs symadd
python main.py "${DIR}" egc zinc --pretrained --hidden 124 --egc-num-heads 4 --egc-num-bases 4 --aggrs add,std,max

# CIFAR
python main.py "${DIR}" gatv2 cifar --pretrained --hidden 104
python main.py "${DIR}" egc cifar --pretrained --hidden 168 --egc-num-heads 8 --egc-num-bases 4 --aggrs symadd
python main.py "${DIR}" egc cifar --pretrained --hidden 128 --egc-num-heads 4 --egc-num-bases 4 --aggrs symadd,std,max

# HIV
python main.py "${DIR}" gcn hiv --pretrained --hidden 240
python main.py "${DIR}" gat hiv --pretrained --hidden 240
python main.py "${DIR}" gatv2 hiv --pretrained --hidden 184
python main.py "${DIR}" gin hiv --pretrained --hidden 240
python main.py "${DIR}" sage hiv --pretrained --hidden 180
python main.py "${DIR}" mpnn-max hiv --pretrained --hidden 180
python main.py "${DIR}" mpnn-sum hiv --pretrained --hidden 180
python main.py "${DIR}" egc hiv --pretrained --hidden 296 --egc-num-heads 8 --egc-num-bases 4 --aggrs symadd
python main.py "${DIR}" egc hiv --pretrained --hidden 224 --egc-num-heads 4 --egc-num-bases 4 --aggrs add,mean,max

# Arxiv
python main.py "${DIR}" gcn arxiv --pretrained --hidden 156
python main.py "${DIR}" gat arxiv --pretrained --hidden 152
python main.py "${DIR}" gatv2 arxiv --pretrained --hidden 112
python main.py "${DIR}" gin arxiv --pretrained --hidden 156
python main.py "${DIR}" sage arxiv --pretrained --hidden 115
python main.py "${DIR}" mpnn-max arxiv --pretrained --hidden 116
python main.py "${DIR}" mpnn-sum arxiv --pretrained --hidden 116
python main.py "${DIR}" pna arxiv --pretrained --hidden 76
python main.py "${DIR}" egc arxiv --pretrained --hidden 184 --egc-num-heads 8 --egc-num-bases 4 --aggrs symadd
python main.py "${DIR}" egc arxiv --pretrained --hidden 136 --egc-num-heads 4 --egc-num-bases 4 --aggrs symadd,max,mean

# Code2
python main.py "${DIR}" gcn code --pretrained --hidden 304
python main.py "${DIR}" gat code --pretrained --hidden 304
python main.py "${DIR}" gatv2 code --pretrained --hidden 296
python main.py "${DIR}" gin code --pretrained --hidden 304
python main.py "${DIR}" sage code --pretrained --hidden 293
python main.py "${DIR}" mpnn-max code --pretrained --hidden 292
python main.py "${DIR}" mpnn-sum code --pretrained --hidden 292
python main.py "${DIR}" pna code --pretrained --hidden 272
python main.py "${DIR}" egc code --pretrained --hidden 304 --egc-num-heads 8 --egc-num-bases 8 --aggrs symadd
python main.py "${DIR}" egc code --pretrained --hidden 300 --egc-num-heads 4 --egc-num-bases 4 --aggrs symadd,min,max
