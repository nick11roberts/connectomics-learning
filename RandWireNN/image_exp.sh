#!/bin/bash

### 4/13 ###
# python3.6 trainer.py -c config/imagenette_fair.yaml -m imagenette_fair > txt_logs/imagenette_fair.log
# python3.6 trainer.py -c config/imagenette_g_highest.yaml -m imagenette_g_highest > txt_logs/imagenette_g_highest.log
# python3.6 trainer.py -c config/imagenette_g_lowest.yaml -m imagenette_g_lowest > txt_logs/imagenette_g_lowest.log
# python3.6 trainer.py -c config/imagenette_g_mean.yaml -m imagenette_g_mean > txt_logs/imagenette_g_mean.log

# python3.6 trainer.py -c config/imagewoof_fair.yaml -m imagewoof_fair > txt_logs/imagewoof_fair.log
# python3.6 trainer.py -c config/imagewoof_g_highest.yaml -m imagewoof_g_highest > txt_logs/imagewoof_g_highest.log
# python3.6 trainer.py -c config/imagewoof_g_lowest.yaml -m imagewoof_g_lowest > txt_logs/imagewoof_g_lowest.log
# python3.6 trainer.py -c config/imagewoof_g_mean.yaml -m imagewoof_g_mean > txt_logs/imagewoof_g_mean.log

### 4/14 ###
# python3.6 trainer.py -c config/imagenette_g_368801.yaml -m imagenette_g_368801 > txt_logs/imagenette_g_368801.log
# python3.6 trainer.py -c config/imagenette_g_3095.yaml -m imagenette_g_3095 > txt_logs/imagenette_g_3095.log
# python3.6 trainer.py -c config/imagenette_g_346433.yaml -m imagenette_g_346433 > txt_logs/imagenette_g_346433.log

# python3.6 trainer.py -c config/imagewoof_g_368801.yaml -m imagewoof_g_368801 > txt_logs/imagewoof_g_368801.log
# python3.6 trainer.py -c config/imagewoof_g_3095.yaml -m imagewoof_g_3095 > txt_logs/imagewoof_g_3095.log
# python3.6 trainer.py -c config/imagewoof_g_346433.yaml -m imagewoof_g_346433 > txt_logs/imagewoof_g_346433.log

### 4/15 ###
# python3.6 trainer.py -c config/imagenette_g_lowest_noram.yaml -m imagenette_g_lowest_noram
# python3.6 trainer.py -c config/imagenette_g_r_lambdaA_min.yaml -m imagenette_g_r_lambdaA_min
# python3.6 trainer.py -c config/imagenette_g_r_lambdaA_max.yaml -m imagenette_g_r_lambdaA_max

# python3.6 trainer.py -c config/imagewoof_g_lowest_noram.yaml -m imagewoof_g_lowest_noram
# python3.6 trainer.py -c config/imagewoof_g_r_lambdaA_min.yaml -m imagewoof_g_r_lambdaA_min
# python3.6 trainer.py -c config/imagewoof_g_r_lambdaA_max.yaml -m imagewoof_g_r_lambdaA_max

### 4/16 ### (re-ran all experiments but with epoch-wise evaluation)
# python3.6 trainer.py -c config/imagenette_g_lowest_noram.yaml -m imagenette_g_lowest_noram
# python3.6 trainer.py -c config/imagenette_g_r_lambdaA_min.yaml -m imagenette_g_r_lambdaA_min
# python3.6 trainer.py -c config/imagenette_g_r_lambdaA_max.yaml -m imagenette_g_r_lambdaA_max
# python3.6 trainer.py -c config/imagenette_fair.yaml -m imagenette_fair
# python3.6 trainer.py -c config/imagenette_fair_repl.yaml -m imagenette_fair_repl

# python3.6 trainer.py -c config/imagewoof_g_lowest_noram.yaml -m imagewoof_g_lowest_noram
# python3.6 trainer.py -c config/imagewoof_g_r_lambdaA_min.yaml -m imagewoof_g_r_lambdaA_min
# python3.6 trainer.py -c config/imagewoof_g_r_lambdaA_max.yaml -m imagewoof_g_r_lambdaA_max
# python3.6 trainer.py -c config/imagewoof_fair.yaml -m imagewoof_fair
# python3.6 trainer.py -c config/imagewoof_fair_repl.yaml -m imagewoof_fair_repl


# python3.6 trainer.py -c config/imagenette_g_highest.yaml -m imagenette_g_highest
# python3.6 trainer.py -c config/imagenette_g_lowest.yaml -m imagenette_g_lowest
# python3.6 trainer.py -c config/imagenette_g_mean.yaml -m imagenette_g_mean

# python3.6 trainer.py -c config/imagewoof_g_highest.yaml -m imagewoof_g_highest
# python3.6 trainer.py -c config/imagewoof_g_lowest.yaml -m imagewoof_g_lowest
# python3.6 trainer.py -c config/imagewoof_g_mean.yaml -m imagewoof_g_mean

# python3.6 trainer.py -c config/imagenette_g_368801.yaml -m imagenette_g_368801
# python3.6 trainer.py -c config/imagenette_g_3095.yaml -m imagenette_g_3095
# python3.6 trainer.py -c config/imagenette_g_346433.yaml -m imagenette_g_346433

# python3.6 trainer.py -c config/imagewoof_g_368801.yaml -m imagewoof_g_368801
# python3.6 trainer.py -c config/imagewoof_g_3095.yaml -m imagewoof_g_3095
# python3.6 trainer.py -c config/imagewoof_g_346433.yaml -m imagewoof_g_346433


### 4/17 ### (All experiments, but on MNIST)
# python3.6 trainer_mnist.py -c config/imagenette_g_lowest_noram.yaml -m mnist_g_lowest_noram_2
# python3.6 trainer_mnist.py -c config/imagenette_g_r_lambdaA_min.yaml -m mnist_g_r_lambdaA_min_2
# python3.6 trainer_mnist.py -c config/imagenette_g_r_lambdaA_max.yaml -m mnist_g_r_lambdaA_max_2
# python3.6 trainer_mnist.py -c config/imagenette_fair.yaml -m mnist_fair_2
# python3.6 trainer_mnist.py -c config/imagenette_fair_repl.yaml -m mnist_fair_repl_2

# python3.6 trainer_mnist.py -c config/imagenette_g_highest.yaml -m mnist_g_highest_2
# python3.6 trainer_mnist.py -c config/imagenette_g_lowest.yaml -m mnist_g_lowest_2
# python3.6 trainer_mnist.py -c config/imagenette_g_mean.yaml -m mnist_g_mean_2

# python3.6 trainer_mnist.py -c config/imagenette_g_368801.yaml -m mnist_g_368801_2
# python3.6 trainer_mnist.py -c config/imagenette_g_3095.yaml -m mnist_g_3095_2
# python3.6 trainer_mnist.py -c config/imagenette_g_346433.yaml -m mnist_g_346433_2

### 4/22 ### (All experiments, but on Fashion MNIST) NOTE: Accidentally overwrote mnist_fair TODO redo
# python3.6 trainer_fashion_mnist.py -c config/imagenette_g_lowest_noram.yaml -m fashion_mnist_g_lowest_noram_2
# python3.6 trainer_fashion_mnist.py -c config/imagenette_g_r_lambdaA_min.yaml -m fashion_mnist_g_r_lambdaA_min_2
# python3.6 trainer_fashion_mnist.py -c config/imagenette_g_r_lambdaA_max.yaml -m fashion_mnist_g_r_lambdaA_max_2
# python3.6 trainer_fashion_mnist.py -c config/imagenette_fair.yaml -m fashion_mnist_fair_2
# python3.6 trainer_fashion_mnist.py -c config/imagenette_fair_repl.yaml -m fashion_mnist_fair_repl_2

# python3.6 trainer_fashion_mnist.py -c config/imagenette_g_highest.yaml -m fashion_mnist_g_highest_2
# python3.6 trainer_fashion_mnist.py -c config/imagenette_g_lowest.yaml -m fashion_mnist_g_lowest_2
# python3.6 trainer_fashion_mnist.py -c config/imagenette_g_mean.yaml -m fashion_mnist_g_mean_2

# python3.6 trainer_fashion_mnist.py -c config/imagenette_g_368801.yaml -m fashion_mnist_g_368801_2
# python3.6 trainer_fashion_mnist.py -c config/imagenette_g_3095.yaml -m fashion_mnist_g_3095_2
# python3.6 trainer_fashion_mnist.py -c config/imagenette_g_346433.yaml -m fashion_mnist_g_346433_2

# python3.6 trainer_fashion_mnist.py -c config/imagenette_g_lowest_noram.yaml -m longer_fashion_mnist_g_lowest_noram
# python3.6 trainer_fashion_mnist.py -c config/imagenette_g_r_lambdaA_min.yaml -m longer_fashion_mnist_g_r_lambdaA_min
# python3.6 trainer_fashion_mnist.py -c config/imagenette_g_r_lambdaA_max.yaml -m longer_fashion_mnist_g_r_lambdaA_max
# python3.6 trainer_fashion_mnist.py -c config/imagenette_fair.yaml -m longer_fashion_mnist_fair
# python3.6 trainer_fashion_mnist.py -c config/imagenette_fair_repl.yaml -m longer_fashion_mnist_fair_repl

# python3.6 trainer_fashion_mnist.py -c config/imagenette_g_highest.yaml -m longer_fashion_mnist_g_highest
# python3.6 trainer_fashion_mnist.py -c config/imagenette_g_lowest.yaml -m longer_fashion_mnist_g_lowest
# python3.6 trainer_fashion_mnist.py -c config/imagenette_g_mean.yaml -m longer_fashion_mnist_g_mean

# python3.6 trainer_fashion_mnist.py -c config/imagenette_g_368801.yaml -m longer_fashion_mnist_g_368801
# python3.6 trainer_fashion_mnist.py -c config/imagenette_g_3095.yaml -m longer_fashion_mnist_g_3095
# python3.6 trainer_fashion_mnist.py -c config/imagenette_g_346433.yaml -m longer_fashion_mnist_g_346433

# python3.6 trainer_mnist.py -c config/imagenette_g_lowest_noram.yaml -m mnist_g_lowest_noram
# python3.6 trainer_mnist.py -c config/imagenette_g_r_lambdaA_min.yaml -m mnist_g_r_lambdaA_min
# python3.6 trainer_mnist.py -c config/imagenette_g_r_lambdaA_max.yaml -m mnist_g_r_lambdaA_max
# python3.6 trainer_mnist.py -c config/imagenette_fair.yaml -m mnist_fair
# python3.6 trainer_mnist.py -c config/imagenette_fair_repl.yaml -m mnist_fair_repl
# python3.6 trainer_mnist.py -c config/imagenette_g_highest.yaml -m mnist_g_highest
# python3.6 trainer_mnist.py -c config/imagenette_g_lowest.yaml -m mnist_g_lowest
# python3.6 trainer_mnist.py -c config/imagenette_g_mean.yaml -m mnist_g_mean
# python3.6 trainer_mnist.py -c config/imagenette_g_368801.yaml -m mnist_g_368801
# python3.6 trainer_mnist.py -c config/imagenette_g_3095.yaml -m mnist_g_3095
# python3.6 trainer_mnist.py -c config/imagenette_g_346433.yaml -m mnist_g_346433
# python3.6 trainer_fashion_mnist.py -c config/imagenette_g_lowest_noram.yaml -m fashion_mnist_g_lowest_noram
# python3.6 trainer_fashion_mnist.py -c config/imagenette_g_r_lambdaA_min.yaml -m fashion_mnist_g_r_lambdaA_min
# python3.6 trainer_fashion_mnist.py -c config/imagenette_g_r_lambdaA_max.yaml -m fashion_mnist_g_r_lambdaA_max
# python3.6 trainer_fashion_mnist.py -c config/imagenette_fair.yaml -m fashion_mnist_fair
# python3.6 trainer_fashion_mnist.py -c config/imagenette_fair_repl.yaml -m fashion_mnist_fair_repl
# python3.6 trainer_fashion_mnist.py -c config/imagenette_g_highest.yaml -m fashion_mnist_g_highest
# python3.6 trainer_fashion_mnist.py -c config/imagenette_g_lowest.yaml -m fashion_mnist_g_lowest
# python3.6 trainer_fashion_mnist.py -c config/imagenette_g_mean.yaml -m fashion_mnist_g_mean
# python3.6 trainer_fashion_mnist.py -c config/imagenette_g_368801.yaml -m fashion_mnist_g_368801
# python3.6 trainer_fashion_mnist.py -c config/imagenette_g_3095.yaml -m fashion_mnist_g_3095
# python3.6 trainer_fashion_mnist.py -c config/imagenette_g_346433.yaml -m fashion_mnist_g_346433

python3.6 trainer_mnist.py -c config/imagenette_g_lowest_noram.yaml -m mnist_g_lowest_noram_2
python3.6 trainer_mnist.py -c config/imagenette_g_r_lambdaA_min.yaml -m mnist_g_r_lambdaA_min_2
python3.6 trainer_mnist.py -c config/imagenette_g_r_lambdaA_max.yaml -m mnist_g_r_lambdaA_max_2
python3.6 trainer_mnist.py -c config/imagenette_fair.yaml -m mnist_fair_2
python3.6 trainer_mnist.py -c config/imagenette_fair_repl.yaml -m mnist_fair_repl_2
python3.6 trainer_mnist.py -c config/imagenette_g_highest.yaml -m mnist_g_highest_2
python3.6 trainer_mnist.py -c config/imagenette_g_lowest.yaml -m mnist_g_lowest_2
python3.6 trainer_mnist.py -c config/imagenette_g_mean.yaml -m mnist_g_mean_2
python3.6 trainer_mnist.py -c config/imagenette_g_368801.yaml -m mnist_g_368801_2
python3.6 trainer_mnist.py -c config/imagenette_g_3095.yaml -m mnist_g_3095_2
python3.6 trainer_mnist.py -c config/imagenette_g_346433.yaml -m mnist_g_346433_2
python3.6 trainer_fashion_mnist.py -c config/imagenette_g_lowest_noram.yaml -m fashion_mnist_g_lowest_noram_2
python3.6 trainer_fashion_mnist.py -c config/imagenette_g_r_lambdaA_min.yaml -m fashion_mnist_g_r_lambdaA_min_2
python3.6 trainer_fashion_mnist.py -c config/imagenette_g_r_lambdaA_max.yaml -m fashion_mnist_g_r_lambdaA_max_2
python3.6 trainer_fashion_mnist.py -c config/imagenette_fair.yaml -m fashion_mnist_fair_2
python3.6 trainer_fashion_mnist.py -c config/imagenette_fair_repl.yaml -m fashion_mnist_fair_repl_2
python3.6 trainer_fashion_mnist.py -c config/imagenette_g_highest.yaml -m fashion_mnist_g_highest_2
python3.6 trainer_fashion_mnist.py -c config/imagenette_g_lowest.yaml -m fashion_mnist_g_lowest_2
python3.6 trainer_fashion_mnist.py -c config/imagenette_g_mean.yaml -m fashion_mnist_g_mean_2
python3.6 trainer_fashion_mnist.py -c config/imagenette_g_368801.yaml -m fashion_mnist_g_368801_2
python3.6 trainer_fashion_mnist.py -c config/imagenette_g_3095.yaml -m fashion_mnist_g_3095_2
python3.6 trainer_fashion_mnist.py -c config/imagenette_g_346433.yaml -m fashion_mnist_g_346433_2