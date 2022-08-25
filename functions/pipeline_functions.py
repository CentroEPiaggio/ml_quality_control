import random
import tensorflow as tf
import tensorflow_addons as tfa
from pathlib import Path

def sample_frames(x, N, parent_path):

    """
    Helper function to get a list of frames from a path and sample N frames out of it.
    """

    #Sample in a proportional way depending on the number of frames available:
    frames = sorted([(str(f), *x[1:]) for f in Path(parent_path+x[0]).iterdir() if f.is_file() and f.suffix in ['.jpg', '.png']])
    n_frames = len(frames)

    if n_frames >= 700:
        N = 2*N
    
    return frames[-N:]

def get_frame_ls(video_records, hypers, parent_path):

    """
    Function to get a list of frames (path + other info) from the video records.

    Args:
        video_records (list): list of tuples in the format (video_id, class, add_data)
        hypers (dict): the configuration dictionary
        parent_path (str): the path to the folder containing the videos
    
    Returns:
        frame_ls (list): list of tuples in the format (frame_path, class, add_data)
    """

    #Unpack the parameters:
    N = hypers['N']

    #Get all frames and sample the last N frames from the list:
    frame_ls = [sample_frames(x, N, parent_path) for x in video_records]
    #and flatten the output into a list of frames:
    frame_ls = [y for x in frame_ls for y in x]

    return frame_ls

def map_label(x):

    """
    Helper function to manually one-hot encode the labels.
    """

    if str(x) == 'ok':
        lb = [1, 0, 0]
    elif str(x) == 'over_e':
        lb = [0, 1, 0]
    elif str(x) == 'under_e':
        lb = [0, 0, 1]
    else:
        lb = [0, 0, 0]

    return lb

def parse_frames_gray(path_tensor, hypers):

    """
    Load a frame in memory (as a single gray imae) and apply some pre-processing functions.

    Args:
        path_tensor (tensorflow.tensor): string tensor containing the path to the image
        hypers (dict): configuration values

    Returns:
        frame (tensorflow.tensor): float32 tensor containing the image
    """

    #Unpack the parameters:
    input_shape = hypers['input_shape']

    #Read, resize and rescale in range [0, 1]:
    frame = tf.io.decode_jpeg(tf.io.read_file(path_tensor))
    frame = tf.image.resize(frame, input_shape[0:2])
    frame = rescale_min_max(frame)

    return frame

def rescale_min_max(x):
    
    """
    Helper function to rescale img values between 0 and 1.
    """

    return tf.divide(tf.subtract(x, tf.reduce_min(x)), tf.subtract(tf.reduce_max(x), tf.reduce_min(x)))