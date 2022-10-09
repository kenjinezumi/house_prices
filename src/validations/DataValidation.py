import os
from pathlib import Path
import pandas as pd
from DataValidationInterface import DataValidationInterface
import logging
import json

logging.basicConfig(level=logging.INFO)


class DataValidation(DataValidationInterface):

    def __init__(self):
        config = open('./config/config.json')
        config = json.load(config)
        self.data_directory = config["variables"]["data_path"]
        self.test_file_path = config["variables"]["test_path"]
        self.train_file_path = config["variables"]["train_path"]
        self.target_variable = config["variables"]["target_variable"]
        self.check_files_exist()
        self.train_data_set = pd.read_csv(self.train_file_path).\
            drop(columns=self.target_variable, index=1)
        self.test_data_set = pd.read_csv(self.test_file_path)

        logging.info(
            f"\nLocated the test_file_path: {self.test_file_path}\n"
            f"Located the train_file_path: {self.train_file_path}\n"
            f"Located the data_directory: {self.data_directory}\n"
        )

    def check_files_exist(self) -> bool:
        """Check files exists in the data report"""
        print(Path(self.data_directory))
        if not Path(self.data_directory).is_dir():
            logging.error(f"Could not locate the data directory at {self.data_directory}")
            raise Exception(
                f"The directory {self.data_directory} does not exist"
            )
        if not Path(self.test_file_path).exists():
            raise Exception(
                f"The file {self.test_file_path} does not exist"
            )
        if not Path(self.train_file_path).exists():
            raise Exception(
                f"The file {self.train_file_path} does not exist"
            )
        else:
            logging.info("The check of file has been successful!")
        return True

    def check_features_not_missing(self) -> bool:
        """Check features are not missing between two datasets"""
        list_columns_train = sorted(self.train_data_set.columns.values.tolist())
        list_columns_test = sorted(self.test_data_set.columns.values.tolist())
        print("LIST COLUMN TRAIN", list_columns_train)
        if list_columns_test == list_columns_train:
            logging.info("No feature is missing!")
            return True
        else:
            difference = set( list_columns_train).symmetric_difference(set(list_columns_test))
            raise Exception(
                f"The features {difference} are different."
            )
            return False

    def check_dtypes(self) -> bool:
        """Check dtypes of datasets are similar"""
        list_columns_train = sorted(self.train_data_set.columns.values.tolist())
        list_diff = []
        for each in range(len(list_columns_train)):
            value = list_columns_train[each]
            if self.train_data_set[list_columns_train[each]].dtype != \
                    self.test_data_set[list_columns_train[each]].dtype:
                list_diff.append(value)
        if not list_diff:
            return True
        else:
            logging.warning(
                f"The dtypes between the two datasets are different"
                f" for the following columns {list_diff}"
            )
            return True

    def check_duplicated_records(self) -> bool:
        """Check for any duplicated records in the datasets"""
        count_duplicated = sum(self.test_data_set.duplicated().tolist())
        if count_duplicated > 0:
            logging.warning(f"{count_duplicated} duplicates have been spotted")
            #For the time being we'll leave like that but we can remove the duplicates too
            bool_series = self.test_data_set.duplicated(keep='first')
            return True

    def check_missing_data(self) -> bool:
        """Check for any missing data in the datasets"""
        count_null_test = self.test_data_set.isnull().sum()
        if count_null_test > 0:
            logging.warning(f"There's {count_null_test} missing data")
            return True

    def check_missing_data_per_feature(self) -> None:
        """Check missing data per specific value e.g. stirngs or others"""
        results = {}
        for column_feature in set(self.test_data_set):
            count = 0
            count += (self.test_data_set[column_feature] == 'na').sum()
            count += (self.test_data_set[column_feature] == 'NA').sum()
            count += (self.test_data_set[column_feature] == 'Na').sum()
            count += self.test_data_set[column_feature].isnull().sum()
            results[column_feature] = count
        df = pd.DataFrame([results], columns=results.keys()).T
        df.rename(columns={df.columns[0]: "missing_data"}, inplace=True)
        df.to_csv(os.path.join("report/", "missing_data.csv"))



if __name__ == "__main__":
    logging.info("Starting the script")
    data = DataValidation()
    data.check_features_not_missing()
    data.check_dtypes()
    data.check_duplicated_records()
    data.check_missing_data_per_feature()
