import random
import tensorflow as tf

#Import py files:
import sys
sys.path.append('./')
from functions.pipeline_functions import *

def get_dataset(video_records, hypers, augm_flag, parent_path='./data/'):

    """
    Function to get a tf dataset from a series of video records.

    Args:
        video_records (list): lost of tuples, where each tuple refers to a single video and has the format (video_id, class, add_data)
        hypers (dict): configuration parameters
        augm_flag (bool): wheter to shuffle the data or not
        parent_path (str): where to look for the videos. Defaults to the data folder

    Returns:
        final_ds (tf.dataset): final dataset containing (already batched), where each element is in the form (img, class, add_data)
    
    Raises:
        ValueError: if the records list is empty
    """

    if len(video_records) == 0:
        raise ValueError('Please provide a non-empty records list')

    #Unpack the parameters:
    batch_size = hypers['bs']
    r_seed = hypers['seed']

    #Get the frames:
    frame_ls = get_frame_ls(video_records, hypers, parent_path)

    if augm_flag:
        random.Random(r_seed).shuffle(frame_ls)

    #Convert to dataset:
    frame_ds = tf.data.Dataset.from_tensor_slices([x[0] for x in frame_ls])
    label_ds = tf.data.Dataset.from_tensor_slices([map_label(x[1]) for x in frame_ls])
    add_ds = tf.data.Dataset.from_tensor_slices([x[2:] for x in frame_ls])

    #Zip together:
    final_ds = tf.data.Dataset.zip((frame_ds, label_ds, add_ds))

    #Load the frames into memory:
    final_ds = final_ds.map(lambda x, y, z: (parse_frames_gray(x, hypers), y, z)).cache()

    final_ds = final_ds.batch(batch_size, drop_remainder=False).prefetch(buffer_size=tf.data.AUTOTUNE)

    return final_ds