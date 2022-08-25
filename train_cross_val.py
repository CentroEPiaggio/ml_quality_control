from datetime import datetime
import json
import logging
from pathlib import Path
import tensorflow as tf

#Import py files:
import sys

sys.path.append('./')
from functions.model_wrapper import ModelWrapper

depth = [5, 6, 7]
models = ['simple', 'vgg', 'resnet']

pars = [(d, m) for d in depth for m in models]
for par in pars:

    print('Starting run with {} depth, {} model...'.format(par[0], par[1]))

    #Create the results directory for this run:
    res_dir = './exp/cross_val/model_d_{}_m_{}'.format(par[0], par[1])
    Path(res_dir).mkdir(parents=True, exist_ok=False)

    #Load config:
    config = json.load(open('./config.json', 'r'))

    #Set the config:
    config['cnn_model']['depth'] = par[0]
    config['cnn_model']['model_conf'] = par[1]

    #Save the config for future reference:
    json.dump(config, open(res_dir+'/config.json', 'w'))

    mw = ModelWrapper(config)
    mw.perform_exp(res_dir, mode_flag='cross_val')

    #Reset session:
    tf.keras.backend.clear_session()
    tf.compat.v1.reset_default_graph()
    del mw
