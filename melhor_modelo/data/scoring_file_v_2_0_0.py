# ---------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# ---------------------------------------------------------
import json
import logging
import os
import pickle
import numpy as np
import pandas as pd
import joblib

import azureml.automl.core
from azureml.automl.core.shared import logging_utilities, log_server
from azureml.telemetry import INSTRUMENTATION_KEY

from inference_schema.schema_decorators import input_schema, output_schema
from inference_schema.parameter_types.numpy_parameter_type import NumpyParameterType
from inference_schema.parameter_types.pandas_parameter_type import PandasParameterType
from inference_schema.parameter_types.standard_py_parameter_type import StandardPythonParameterType

data_sample = PandasParameterType(pd.DataFrame({"Player": pd.Series(["example_value"], dtype="object"), "Team": pd.Series(["example_value"], dtype="object"), "Age": pd.Series([0], dtype="int8"), "GP": pd.Series([0], dtype="int8"), "W": pd.Series([0], dtype="int8"), "L": pd.Series([0], dtype="int8"), "Min": pd.Series([0.0], dtype="float32"), "FGM": pd.Series([0.0], dtype="float32"), "FGA": pd.Series([0.0], dtype="float32"), "FG%": pd.Series([0.0], dtype="float32"), "3PM": pd.Series([0.0], dtype="float32"), "3PA": pd.Series([0.0], dtype="float32"), "3P%": pd.Series([0.0], dtype="float32"), "FTM": pd.Series([0.0], dtype="float32"), "FTA": pd.Series([0.0], dtype="float32"), "FT%": pd.Series([0.0], dtype="float32"), "OREB": pd.Series([0.0], dtype="float32"), "DREB": pd.Series([0.0], dtype="float32"), "REB": pd.Series([0.0], dtype="float32"), "AST": pd.Series([0.0], dtype="float32"), "TOV": pd.Series([0.0], dtype="float32"), "STL": pd.Series([0.0], dtype="float32"), "BLK": pd.Series([0.0], dtype="float32"), "PF": pd.Series([0.0], dtype="float32"), "FP": pd.Series([0.0], dtype="float32"), "DD2": pd.Series([0], dtype="int8"), "TD3": pd.Series([0], dtype="int8"), "Plus/Minus": pd.Series([0.0], dtype="float32")}))
input_sample = StandardPythonParameterType({'data': data_sample})

result_sample = NumpyParameterType(np.array([0.0]))
output_sample = StandardPythonParameterType({'Results':result_sample})
sample_global_parameters = StandardPythonParameterType(1.0)

try:
    log_server.enable_telemetry(INSTRUMENTATION_KEY)
    log_server.set_verbosity('INFO')
    logger = logging.getLogger('azureml.automl.core.scoring_script_v2')
except:
    pass


def get_model_root(model_root: str):
    root_contents = os.listdir(model_root)
    logger.info(f"List model root dir: {os.listdir(model_root)}")
    if len(root_contents) == 1:
        root_file_path = os.path.join(model_root, root_contents[0])
        return root_file_path if os.path.isdir(root_file_path) else model_root
    else:
        raise Exception("Unexpected. root must contain a model file or a mlflow model directory")


def init():
    global model
    # This name is model.id of model that we want to deploy deserialize the model file back
    # into a sklearn model
    model_root = get_model_root(os.getenv('AZUREML_MODEL_DIR'))
    model_path = os.path.join(model_root, 'model.pkl')
    path = os.path.normpath(model_path)
    path_split = path.split(os.sep)
    log_server.update_custom_dimensions({'model_name': path_split[-3], 'model_version': path_split[-2]})
    try:
        logger.info("Loading model from path.")
        model = joblib.load(model_path)
        logger.info("Loading successful.")
    except Exception as e:
        logging_utilities.log_traceback(e, logger)
        raise

@input_schema('Inputs', input_sample)
@input_schema('GlobalParameters', sample_global_parameters, convert_to_provided_type=False)
@output_schema(output_sample)
def run(Inputs, GlobalParameters=1.0):
    data = Inputs['data']
    result = model.predict(data)
    return {'Results':result.tolist()}
