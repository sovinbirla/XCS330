#!/bin/bash

# Q1
python3 main.py --task plot_ft --model bert-tiny,bert-med --dataset amazon --k 1,8,128 --output submission/results/q1_plot.png

# Q2
python3 main.py --task plot_icl --model med,full --dataset babi --k 0,1,16 --output submission/results/q2_babi_plot.png
python3 main.py --task plot_icl --model med,full --dataset xsum --k 0,1,4 --prompt none,tldr,custom --output submission/results/q2_xsum_plot.png

# Q3
python3 main.py --task plot_ft --model med --mode first,last,middle,lora4,lora16 --dataset xsum --k 0,1,8,128 --output submission/results/q3_xsum_plot.png
python3 main.py --task plot_ft --model med --mode first,last,middle,lora4,lora16 --dataset babi --k 0,1,8,128 --output submission/results/q3_babi_plot.png

# Q4
python3 main.py --task plot --output submission/results/q4_plot.png