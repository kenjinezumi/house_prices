import numpy as np
import pandas as pd


class DataValidationInterface:

    def __init__(self,
                 data_directory,
                 test_file_path,
                 train_file_path):
        pass

    def check_files_exist(self) -> bool:
        """Check files exists in the data report"""
        pass

    def check_feature_not_missing(self) -> bool:
        """Check features are not missing between two datasets"""
        pass

    def check_dtypes(self) -> bool:
        """Check dtypes of datasets are similar"""
        pass

    def check_missing_data(self) -> bool:
        """Check for any missing data in the datasets"""
        pass

    def check_duplicated_records(self) -> bool:
        """Check for any duplicated records in the datasets"""
        pass

    def check_outliers(self) -> bool:
        """Check for any potential outliers in the datasets"""
        pass


