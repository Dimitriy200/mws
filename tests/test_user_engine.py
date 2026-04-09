# ======================================================
# Тест pipeline
# ======================================================

import pandas as pd
import numpy as np
import logging

import sys
from pathlib import Path
parent_dir = Path(__file__).parent.parent
sys.path.append(str(parent_dir))

from mws.config import (
    PATH_LOG,
    PATH_SKALERS,

    PATH_TRAIN_RAW,
    PATH_TRAIN_ADD_RAW,

    MLFLOW_TRACKING_URI,
    MLFLOW_USERNAME,
    MLFLOW_REPO_OWNER,
    MLFLOW_REPO_NAME,
    MLFLOW_REPO_TOKEN,
    MLFLOW_REPO_PASSWORD
)

from pathlib import Path
from mws.pipeline.pipeline_spec import Pipeline
# from mws.pipeline.pipeline import Pipeline
from mws.preprocessing.scaler import Scaler
from mws.preprocessing.load_data_first import LoadDataTrain
from mws.preprocessing.load_data_add import LoadDataTrainAdd
# from src.training.experiment import Experiment
from mws.training.experiment_new import Experiment
from mws.models.autoencoder import AutoEncoder
from mws.training.trainer import train_model
from mws.training.thresholding import choose_optimal_threshold_stadart, choose_optimal_threshold_un


# ======================================================
# 1 Подготовка Loader и Scaller
# ======================================================
loader = LoadDataTrainAdd()
scaler_manager = Scaler()

pipeline = Pipeline(
    # path_data_dir = PATH_TRAIN_RAW,
    path_data_dir=Path(PATH_TRAIN_ADD_RAW).joinpath("2024-07-02_2024-07-03_2024-07-04"),
    path_scaler=Path(PATH_SKALERS).joinpath("test_skaller.pkl"),
    scaler_manager=scaler_manager,
    loader=loader
    )

# ======================================================
# 2 Предобработка данных
# ======================================================
final_dataframes = pipeline.run_new()

# ======================================================
# 3 Проведение эксперимента
# ======================================================

experiment = Experiment(
    mlflow_tracking_uri=MLFLOW_TRACKING_URI,
    mlflow_repo_owner=MLFLOW_REPO_OWNER,
    mlflow_repo_name=MLFLOW_REPO_NAME,
    mlflow_username=MLFLOW_USERNAME,
    mlflow_pass=MLFLOW_REPO_PASSWORD,
    mlflow_token=MLFLOW_REPO_TOKEN,
    train_data=final_dataframes,

    model_name = "ae_spec_ver",
    experiment_name = "AE_spec_ver"
)

encoder = experiment.load_model_from_mlflow()
# encoder = AutoEncoder()

# fit_data = encoder.fit(
#     X_train=final_dataframes['X_train'],
#     X_test=final_dataframes['X_test'],
# )
history = encoder.fit(
    x=final_dataframes['X_train'],
    y=final_dataframes['X_train'],
    epochs=5,
    batch_size=80
)

fit_data = {
    "model": encoder,
    "history": history.history
}
# results_threshold = choose_optimal_threshold_un(
#     model=fit_data['model'],
#     X_val=final_dataframes['X_val'],
#     y_val=final_dataframes['y_val']
# )

run_id = experiment.send_experiment_to_mlflow_mini(
    model=fit_data['model'],
    # threshold_result=results_threshold,
    training_history=fit_data['history'],
    # split_data=final_dataframes
)