import json
from pathlib import Path

#Import py files:
from functions.model_wrapper import ModelWrapper

#To run this experiment, the configuration file should be set to: model_conf = 'simple' and depth = 6!
print('Starting run with 6 depth, simple models...')

res_dir = './exp/retrain/'
Path(res_dir).mkdir(parents=True, exist_ok=False)

#Load config:
config = json.load(open('./config.json', 'r'))

#Save the config for future reference:
json.dump(config, open(res_dir+'/config.json', 'w'))

mw = ModelWrapper(config)
mw.perform_exp(res_dir, mode_flag='retrain')