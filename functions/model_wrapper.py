import logging
import pandas as pd
import numpy as np
from statistics import mode
from sklearn.metrics import classification_report, confusion_matrix
import time

#Import py files:
import sys

sys.path.append('./')

from functions import gen_pipeline
from functions.cnn import CNN
from functions.data_wrapper import DataWrapper
from functions.misc_func import unpack_label

import tensorflow as tf
tf.config.optimizer.set_experimental_options({'layout_optimizer': False})
tf.keras.mixed_precision.set_global_policy('mixed_float16')
tf.get_logger().setLevel('ERROR')

class ModelWrapper():

    """Top level class to train and evaluate the CNN for the quality control loop in EBB."""  

    def __init__(self, config):

        #Unpack the hyperparameters for better readability:
        self.config = config
        self.shared = config['shared']

        #Init the wrappers:
        self.cw = CNN(config)
        self.dw = DataWrapper(config)
        
    def perform_exp(self, res_dir, mode_flag='split'):

        """
        Train and evaluate the current model.
        """

        #Set seed:
        self.set_seed()

        #Experiment logging:
        logging.basicConfig(filename=res_dir+'/experiment.log', format='%(levelname)s: %(message)s', level=logging.INFO, force=True)
        logging.getLogger().addHandler(logging.StreamHandler(sys.stdout))
        logging.info('Starting an experiment...')

        #-----------------------------------
        # Training
        #-----------------------------------

        logging.info('Training a new classifier model...')

        try:
            self.cw.init_model()
            self.cw.train_model(res_dir, self.dw.df_train, df_test=self.dw.df_test, mode_flag=mode_flag)
        except Exception as e:
            logging.exception(e)

        #-----------------------------------
        # Evaluation
        #-----------------------------------

        if mode_flag == 'retrain':

            try:
                #Eval:
                self.cw.eval_model(self.dw.df_test)

                #Save the best model:
                self.cw.model.save(res_dir+'/best_model/')
            except Exception as e:
                logging.exception(e)
    
    def set_seed(self):

        """
        Helper function to globally set the random seed.
        Taken from: https://odsc.medium.com/properly-setting-the-random-seed-in-ml-experiments-not-as-simple-as-you-might-imagine-219969c84752
        """

        #Unpack hypers:
        r_seed = self.shared['seed']

        import random
        random.seed(r_seed)

        import tensorflow as tf
        tf.keras.utils.set_random_seed(r_seed)


if __name__ == '__main__':
    
    import json
    config = json.load(open('./config.json', 'r'))

    mw = ModelWrapper(config)
    mw.perform_exp('./')