def unpack_label(x):

    if x == 'ok':
        lb = 0
    elif x == 'over_e':
        lb = 1
    elif x == 'under_e':
        lb = 2
    
    return lb

def format_records(df, shared):

    """
    Function to format the video dataframe to a list of tuples for the data pipeline.

    Args:
        df (pandas.DataFrame): the dataframe to format
        shared (dict): configuration dictionary

    Returns:
        res (list): list of tuples in the format (video_id, class, add_data)
    """

    res = list(df[['video_id', 'class', *shared['add_data']]].itertuples(index=False, name=None))
    return res

def validate_conf(conf):

    """
    Helper function to check that all required inputs are present in the configuration file.

    Args:
        conf (dict): the configuration dictionary to be validated

    Raises:
        ValueError: if some information is missing
    """

    #List of needed parameters:
    val_shared = [
        'master_path',
        'input_shape',
        'seed',
        'N',
        'bs',
        'add_data'
    ]

    val_model = [
        'model_conf',
        'depth',
        'init_fs',
        'reg_conf',
        'reg_val',
        'lr',
        'epochs',
        'early_stop'
    ]

    #Check the shared subsection:
    for par in val_shared:
        if conf.get('shared').get(par) == None:
            raise ValueError('The configuration file did not pass validation. Please check that shared/'+par+' is present')

    for par in val_model:
        if conf.get('cnn_model').get(par) == None:
            raise ValueError('The configuration file did not pass validation. Please check that cnn_model/'+par+' is present')