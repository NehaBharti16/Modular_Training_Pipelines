import os
from pathlib import Path
import logging

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

project_name = "Training_Pipeline"

list_of_files = [
    #".github/workflows/.gitkeep", # Giving indication to make folder in github which we used in CI/CD deployment
    # Source code structure
    f"src/{project_name}/__init__.py",
    f"src/{project_name}/components/__init__.py",
    f"src/{project_name}/components/data_ingestion.py",
    f"src/{project_name}/components/data_transformation.py",
    f"src/{project_name}/components/model_trainer.py",
    f"src/{project_name}/components/model_monitering.py",
    f"src/{project_name}/pipelines/__init__.py",
    f"src/{project_name}/pipelines/training_pipeline.py",
    f"src/{project_name}/pipelines/prediction_pipeline.py", 
    f"src/{project_name}/exception.py",
    f"src/{project_name}/logger.py",
    f"src/{project_name}/utils.py",
    # Project root files
    "app.py",
    "Dockerfile",
    "setup.py",
    "requirements.txt",

    # Config files structure
    "conf/config.yaml",
    "conf/dataset/california_houses.yaml",
    "conf/dataset/titanic.yaml",
    "conf/task/regression.yaml",
    "conf/task/classification.yaml",
    "conf/model/linear_regression.yaml",
    "conf/model/logistic_regression.yaml",
    "conf/model/random_forest_regressor.yaml",
    "conf/model/random_forest_classifier.yaml",
    "conf/model/gradient_boosting_regressor.yaml",
    "conf/model/gradient_boosting_classifier.yaml",
    "conf/training/default.yaml",
    "conf/logging/mlflow.yaml",
    
]

for filepath in list_of_files:
    filepath = Path(filepath)
    filedir, filename = os.path.split(filepath)
    if filedir != "":
        os.makedirs(filedir, exist_ok=True)
        logging.info(f"Creating directory: {filedir} for the file: {filename}")
    if (not os.path.exists(filepath)) or (os.path.getsize(filepath) == 0):
        with open(filepath, "w") as fp:
            pass
            logging.info(f"Creating empty file: {filepath}")
    else:
        logging.info(f"{filename} File already exists skipping creation.")