from doctest import master
import logging
from typing import final
import pandas as pd

from pathlib import Path
from sklearn.model_selection import train_test_split, StratifiedKFold
from sklearn.utils import resample

#Import py files:
import sys
sys.path.append('./')

from functions import gen_pipeline

class DataWrapper():

    """Class for splitting and preparing the video data for training and evaluation."""

    def __init__(self, hypers):
        
        #Unpack the hyperparameters for better readability:
        self.hypers = hypers
        self.shared = hypers['shared']

        #Get the filtered ds into memory:
        self.master_df = self.get_master_df()

        #Re-sample and split:
        self.df_train, self.df_test = self.split_data(self.master_df.copy())

    def get_master_df(self):

        """
        Function to load the master dataframe into memory and clean it.

        Returns:
            master_df (pandas.DataFrame): master dataframe which contains all the videos information
        """

        #Unpack the parameters:
        master_path = self.shared['master_path']

        #Load the master document:
        master_df = pd.read_csv(master_path).dropna(subset=['video_id', 'class'])

        #Delete the videos with skip_flag:
        master_df = master_df[master_df['skip_flag'] != 1]

        #Get list of videos in the data path:
        videos_ls = sorted([x for x in Path('./data').iterdir() if x.is_dir()])

        #Subset the master document based on the available videos:
        master_df = master_df[master_df['video_id'].isin([str(x.name) for x in videos_ls])]

        #Log at the end:
        logging.info('Found {} videos in the data folder'.format(len(videos_ls)))
        logging.info('Total number of video considered: {}'.format(len(master_df.index)))
        logging.info('Video classes distribution: {}'.format(master_df['class'].value_counts().to_dict()))

        return master_df

    def split_data(self, df):

        """
        Function to get the video splits from a dataframe (with re-sampling).

        Args:
            df (pandas.DataFrame): a dataframe containing the video information

        Returns:
            df_train (pandas.DataFrame): the train dataframe
            df_test (pandas.DataFrame): the test dataframe

        """

        #Unpack the parameters:
        r_seed = self.shared['seed']

        #Balance the dataset by under-sampling to the minority class:
        df = self.re_sample(df)
        logging.info('Using the following video distribution: {}'.format(df['class'].value_counts()))

        #Split:
        df_train, df_test = train_test_split(
            df,
            test_size=0.1,
            stratify=df[['class']],
            random_state=r_seed)

        return df_train, df_test

    def re_sample(self, df):

        """
        Function to randomly re-sample an input dataframe to have class balance.

        Args:
            df (pandas.DataFrame): a dataframe containing the video information
        
        Returns:
            df (pandas.DataFrame): the re-sampled dataframe
        """

        #Unpack the parameters:
        r_seed = self.shared['seed']

        #Under-sample to the lowest value count:
        df, vc = self.del_low_rep(df)
        df = df.groupby('class', as_index=False, group_keys=False).apply(lambda x: x.sample(vc.min(), random_state=r_seed))
        
        return df

    def del_low_rep(self, df, thresh=30):

        """
        Function to delete the low representation classes.

        Args:
            df (pandas.DataFrame): a dataframe containing the video information
            thresh (int): threshold value for the representation. Defaults at 30

        Returns:
            df (pandas.DataFrame): the dataframe without the low representation classes
            vc (pandas.Series): a series containing the value counts for the remaining classes
        
        """

        #Get the value counts:
        vc = df['class'].value_counts()

        #Drop low representation classes (based on count threshold):
        under_rep = vc[vc < thresh].index.to_list()
        df.drop(df[df['class'].isin(under_rep)].index, inplace=True)
        vc.drop(vc[vc < thresh].index, inplace=True)

        return df, vc
    
if __name__ == '__main__':

    import json

    config = json.load(open('./config.json', 'r'))
    dw = DataWrapper(config)

    print('Class distribution for the train videos:')
    print(dw.df_train['class'].value_counts())
    print('Class distribution for the test videos:')
    print(dw.df_test['class'].value_counts())