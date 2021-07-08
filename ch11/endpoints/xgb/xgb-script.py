import os
import xgboost as xgb

def model_fn(model_dir):
    model = xgb.Booster()
    model.load_model(os.path.join(model_dir, 'xgboost-model'))
    return model