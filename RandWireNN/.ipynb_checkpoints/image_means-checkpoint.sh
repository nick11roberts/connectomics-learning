#!/bin/bash

python3.6 trainer.py -c config/imagenette_g_mean.yaml -m imagenette_g_mean > txt_logs/imagenette_g_mean.log
python3.6 trainer.py -c config/imagewoof_g_mean.yaml -m imagewoof_g_mean > txt_logs/imagewoof_g_mean.log

