from audioop import cross
import logging
import tensorflow as tf
from tensorflow.keras import layers, optimizers, models, callbacks, regularizers, Sequential
from keras import backend as K
from keras.utils.layer_utils import count_params

from sklearn.model_selection import KFold, StratifiedKFold, train_test_split
from sklearn.metrics import classification_report, confusion_matrix
import numpy as np

import sys
sys.path.append('./')
from functions.misc_func import format_records
from functions import gen_pipeline

tf.get_logger().setLevel('ERROR')

class CNN():

    """Class to create the CNN model.

    Args:
        config (dict): configuration dictionary containing all necessary parameters
    
    Attributes:
        model (None or keras.Functional): the current model of the class instance, call init to create it!
        config (dict): the current configuration, for better readability also broke down into:
        shared (dict): all shared configuration (seed, shape, etc)
        p_model (dict): all configuration specific to the cnn
    
    """

    def __init__(self, config):

        #Instance of the model:
        self.model = None

        #Config available to the whole class instance:
        self.config = config
        self.shared = config['shared']
        self.p_model = config['cnn_model']

    def init_model(self):

        """
        Function to define different model configurations based on the specified config.
        
        Raises:
            ValueError: if an unknown reg configuration was specified. The knowns are ['l1', 'l2', None]
            ValueError: if an unknown model configuration was specified. The knowns are ['simple', 'vgg', 'resnet']
        """

        #Unpack hyperparameters:
        model_conf = self.p_model['model_conf']
        depth = self.p_model['depth']
        n_filters = self.p_model['init_fs']
        reg_conf = self.p_model['reg_conf']
        reg_val = self.p_model['reg_val']
        lr = self.p_model['lr']
        r_seed = self.shared['seed']

        #Define the regularizer:
        if reg_conf == 'l1':
            reg = regularizers.l1(reg_val)
        elif reg_conf == 'l2':
            reg = regularizers.l2(reg_val)
        elif reg_conf == 'none':
            reg = None
        else:
            raise ValueError('Specified an unknown regularizer')

        #Online image augmentation:
        augm_input = layers.Input(tuple(self.shared['input_shape']))
        xa = layers.RandomFlip('horizontal', seed=r_seed)(augm_input)
        xa = layers.RandomTranslation(height_factor=0.2, width_factor=0.2, seed=r_seed)(xa)
        xa = layers.RandomZoom(0.1, seed=r_seed)(xa)
        xa = layers.RandomContrast(0.2, seed=r_seed)(xa)
        xa = layers.Lambda(lambda x: tf.clip_by_value(x, clip_value_min=0, clip_value_max=1))(xa)
        augm = tf.keras.Model(augm_input, xa, name='augmentation')

        #Convolution blocks:
        if model_conf == 'simple':
            conv = (self.get_simple(depth, n_filters, reg))
        elif model_conf == 'vgg':
            conv = self.get_vgg(depth, n_filters, reg)
        elif model_conf == 'resnet':
            conv = self.get_resnet(depth, n_filters, reg)
        else:
            raise ValueError('Specified an unknown model configuration')

        #Pack the two parts together and add the classification layers:
        augm_repr = augm(augm_input)
        xc = conv(augm_repr)
        xc = layers.GlobalAveragePooling2D()(xc)        
        xc = layers.Dense(3, activation='softmax')(xc)
        self.model = tf.keras.Model(augm_input, xc, name='model')

        #Compile current instance:
        opt = optimizers.Adam(learning_rate=lr)
        self.model.compile(
            optimizer=opt,
            loss='categorical_crossentropy',
            metrics=['categorical_accuracy']
        )
    
    def log_info(self):

        """
        Function to log useful model metrics.
        
        Raises:
            ValueError: if tried to log an empty model
        
        """
        
        if self.model == None:
            raise ValueError('Tried to log an empty model')

        self.model.summary()
        logging.info('Init model with {} parameters'.format(self.model.count_params()))
        logging.info('Number of trainable parameters: {}'.format(count_params(self.model.trainable_weights)))
        logging.info('Number of non trainable parameters: {}'.format(count_params(self.model.non_trainable_weights)))

    def train_model(self, res_dir, df_train, df_test=None, mode_flag='split'):

        """
        Function to train the current model. Three different ways to split the input data are available (based on the mode_flag):

        1. split the train data into train and val for cross-validation
        2. split the train data into train and val using the train-test split
        3. use the test data as validation data (for the final retrain)

        Args:  
            res_dir (str): path to save the tensorboard logs
            df_train (pandas.DataFrame): the train dataframe
            df_test (pandas.DataFrame): the test dataframe. Defaults to None.
            mode_flag (str): how to handle the video data. Defaults to 'split'

        Raises:
            ValueError: if tried to train an empty model
            ValueError: if specified an unknown data handling mode. The knowns are: ['cross_val', 'split', 'retrain']

        """

        if self.model == None:
            raise ValueError('Tried to train an empty model')

        #Unpack hyperparameters:
        epochs = self.p_model['epochs']
        early_stop = self.p_model['early_stop']
        r_seed = self.shared['seed']

        if mode_flag == 'cross_val':
            #Define the splitter and get the indeces:
            cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=r_seed)
            idxs = [(df_train.iloc[train_idx], df_train.iloc[test_idx]) for train_idx, test_idx in cv.split(df_train, df_train['class'])]
        elif mode_flag == 'split':
            #Split into simple train and val:
            train, val = train_test_split(df_train, test_size=0.1111, stratify=df_train['class'], random_state=r_seed)
            idxs = [(train, val)]
        elif mode_flag == 'retrain':
            #Retrain using the test set as 'validation':
            idxs = [(df_train, df_test)]
        else:
            raise ValueError('Specified an unknown data handling mode')

        #Train loop:
        res_acc = []
        res_loss = []
        for fold_idx, el in enumerate(idxs):

            logging.info('Starting training with fold {}/{}...'.format(fold_idx+1, len(idxs)))

            #Reset model at each iteration:
            self.model = None
            self.init_model()

            #Define the callbacks:
            cbs = [
                callbacks.TensorBoard(res_dir+'/logs_clf_{}/'.format(fold_idx), update_freq='batch'),
                callbacks.EarlyStopping('val_loss', patience=early_stop, restore_best_weights=True, verbose=1)
            ]

            #Get the iterators:
            ds_train = gen_pipeline.get_dataset(
                format_records(el[0], self.shared),
                self.shared,
                augm_flag=True
            )

            ds_val = gen_pipeline.get_dataset(
                format_records(el[1], self.shared),
                self.shared,
                augm_flag=False
            )

            #Train:
            self.model.fit(
                ds_train,
                epochs=epochs,
                validation_data=ds_val,
                callbacks=cbs,
                verbose=1
            )

            #Get overall accuracy on the validation data:
            res = self.model.evaluate(ds_val)
            res_acc.append(res[1])
            res_loss.append(res[0])            
            
        logging.info('Validation accuracy on {} folds: {}'.format(len(idxs), res_acc))
        logging.info('Validation loss on {} folds: {}'.format(len(idxs), res_loss))

    def eval_model(self, df):

        """
        Function to perform model evaluation.

        Args:
            df (pandas.DataFrame): a dataframe containing the video information

        Raises:
            ValueError: if called function with an empty model

        """

        if self.model == None:
            raise ValueError('Tried to evaluate an empty model')

        #Get the iterator:
        ds = gen_pipeline.get_dataset(
            format_records(df, self.shared),
            self.shared,
            augm_flag=False
        )

        #Unpack:
        x = np.concatenate(list(ds.map(lambda x, y, z: x)))
        y_true = np.concatenate(list(ds.map(lambda x, y, z: y)))
        y_true = np.argmax(y_true, axis=1)

        #Predict over the x data (class equal to the maximum probability):
        preds = self.model.predict(x)
        y_pred = np.argmax(preds, axis=1)
        
        #Get relevant metrics:
        cm = confusion_matrix(y_true, y_pred)
        cr = classification_report(y_true, y_pred, digits=4)

        logging.info('Confusion matrix:')
        logging.info(cm)
        logging.info('Classification report:')
        logging.info(cr)
    
    def get_simple(self, depth, n_filters, reg):

        """
        Function to get the simple conv model.

        Args:
            depth (int): number of repeating conv blocks
            n_filters (int): number of conv filters for the starting layer (increase by a factor of 2 for the following)
            reg (keras.regularizer): regularizer to be applied to the kernel
        
        Returns:
            simple (keras.Functional): final simple model
        """

        simple_input = layers.Input(tuple(self.shared['input_shape']))
        for i in range(depth):

            if i == 0:
                xs = layers.Conv2D((2**i)*n_filters, (3,3), padding='same', kernel_initializer='he_uniform', kernel_regularizer=reg)(simple_input)
                xs = layers.ReLU()(xs)
            else:
                xs = layers.Conv2D((2**i)*n_filters, (3,3), padding='same', kernel_initializer='he_uniform', kernel_regularizer=reg)(xs)
                xs = layers.ReLU()(xs)
            
            if i != depth-1:
                xs = layers.MaxPool2D()(xs)
            
        simple = tf.keras.Model(simple_input, xs, name='simple')

        return simple
    
    def get_vgg(self, depth, n_filters, reg):

        """
        Function to get the vgg conv model.

        Args:
            depth (int): number of repeating conv blocks
            n_filters (int): number of conv filters for the starting layer (increase by a factor of 2 for the following)
            reg (keras.regularizer): regularizer to be applied to the kernel
        
        Returns:
            vgg (keras.Functional): final vgg model
        """

        vgg_input = layers.Input(tuple(self.shared['input_shape']))
        for i in range(depth):

            if i == 0:
                xv = layers.Conv2D((2**i)*n_filters, (3,3), padding='same', kernel_initializer='he_uniform', kernel_regularizer=reg)(vgg_input)
                xv = layers.ReLU()(xv)
                xv = layers.Conv2D((2**i)*n_filters, (3,3), padding='same', kernel_initializer='he_uniform', kernel_regularizer=reg)(xv)
                xv = layers.ReLU()(xv)
            else:
                xv = layers.Conv2D((2**i)*n_filters, (3,3), padding='same', kernel_initializer='he_uniform', kernel_regularizer=reg)(xv)
                xv = layers.ReLU()(xv)
                xv = layers.Conv2D((2**i)*n_filters, (3,3), padding='same', kernel_initializer='he_uniform', kernel_regularizer=reg)(xv)
                xv = layers.ReLU()(xv)
            
            if i != depth-1:
                xv = layers.MaxPool2D()(xv)
        
        vgg = tf.keras.Model(vgg_input, xv, name='vgg')
        return vgg

    def get_resnet(self, depth, n_filters, reg):

        """
        Function to get the resnet conv model.

        Args:
            depth (int): number of repeating conv blocks
            n_filters (int): number of conv filters for the starting layer (increase by a factor of 2 for the following)
            reg (keras.regularizer): regularizer to be applied to the kernel
        
        Returns:
            resnet (keras.Functional): final resnet model
        """

        resnet_input = layers.Input(tuple(self.shared['input_shape']))
        for i in range(depth):
            if i == 0:
                xr = self.residual_block(resnet_input, (2**i)*n_filters, None)
            elif i <= depth-3:
                xr = self.residual_block(xr, (2**i)*n_filters, None)
            else:
                xr = self.residual_block(xr, (2**i)*n_filters, reg)

        resnet = tf.keras.Model(resnet_input, xr, name='resnet')
        return resnet

    def residual_block(self, x_old, n_filters, reg):

        """
        Function to get a residual block for the resnet model.

        Args:
            x_old (keras.layer): input to the residual block
            n_filters (int): number of conv filters for the layers
            reg (keras.regularizer): regularizer to be applied to the kernel
        
        Returns:
            x_new (keras.Functional): final residual block
        """

        #First two convolutions:
        x_new = layers.Conv2D(n_filters, (3, 3), padding='same', strides=(2, 2), kernel_initializer='he_uniform', kernel_regularizer=reg)(x_old)
        x_new = layers.ReLU()(x_new)

        x_new = layers.Conv2D(n_filters, (3, 3), padding='same', kernel_initializer='he_uniform', kernel_regularizer=reg)(x_new)
        
        #Skip connection:
        x_temp = layers.Conv2D(n_filters, (1, 1), padding='same', strides=(2, 2), kernel_initializer='he_uniform', kernel_regularizer=reg)(x_old) 

        #Sum:
        x_new = layers.Add()([x_temp, x_new])

        #Activation:
        x_new = layers.ReLU()(x_new)

        return x_new

    def get_layer_output(self, img, l_name='augmentation'):

        """
        Function to get the output of a sub-model/layer. Useful for debugging.

        Args:
            img (np.array): image input to the model. The image should be in the shape (1, H, W, 1)
            l_name (str): name of the sub-model/layer we want to acces the output. Defaults to 'augmentation'

        Returns:
            layer_out (np.array): the output of the sub-model/layer
            
        """

        #Get the sub-model/layer by the specified name:
        m = self.model.get_layer(l_name)

        #Set training mode (True to enable data augmentation):
        K.set_learning_phase(True)

        #Function to get the sub-model/layer output:
        functor = K.function([m.layers[0].input], [m.layers[-1].output])

        #Pass image:
        layer_outs = functor([img])

        return layer_outs

if __name__ == '__main__':

    import json

    config = json.load(open('./config.json', 'r'))
    cw = CNN(config)
    cw.init_model()
    cw.log_info()