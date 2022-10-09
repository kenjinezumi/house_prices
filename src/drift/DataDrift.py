import numpy as np
import os
import pandas as pd
from metrics.population_stability_index import  calculate_psi
import logging
import json

logging.basicConfig(level=logging.INFO)


class DataDrift():

    def __init__(self):
        config = open('./config/config.json')
        config = json.load(config)
        self.data_directory = config["variables"]["data_path"]
        self.test_file_path = config["variables"]["test_path"]
        self.train_file_path = config["variables"]["train_path"]
        self.target_variable = config["variables"]["target_variable"]
        self.train_data_set = pd.read_csv(self.train_file_path). \
            drop(columns=self.target_variable, index=1)
        self.test_data_set = pd.read_csv(self.test_file_path)

    def check_drift(self) -> bool:
        """Run population stability index analysis"""
        results_drift = {}
        for column_feature in set(self.train_data_set):
            logging.info(f"Running the drift analysis on {column_feature}")

            if self.train_data_set[column_feature].dtype in (
                    "float64",
                    "int64",
                    "float32",
                    "int32",
            ):
                    logging.info(
                        f"The feature {column_feature} is continuous."
                    )

                    psi = calculate_psi(
                       self.train_data_set[column_feature].to_numpy(),
                       self.test_data_set[column_feature].to_numpy()
                    )
                    results_drift[column_feature] = psi

            else:
                logging.info(
                    f"The feature {column_feature} is categorical."
                )

                psi = calculate_psi(
                    pd.get_dummies(self.train_data_set[column_feature]).to_numpy(),
                    pd.get_dummies(self.test_data_set[column_feature]).to_numpy()
                )
                results_drift[column_feature] = psi[~np.isnan(psi)].mean()
        logging.info(results_drift)
        df = pd.DataFrame([results_drift], columns=results_drift.keys()).T
        df.rename(columns={df.columns[0]: "data_drift"}, inplace=True)
        df.to_csv(os.path.join("report/", "data_drift.csv"))
        return results_drift

if __name__ == "__main__":
    logging.info("Starting the script")
    DataDrift().check_drift()
