import pandas as pd
import logging
import os
from sklearn.model_selection import train_test_split
from step0_utility_functions import Utility


logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

Utility().create_folder('Logs')
params = Utility().read_params()

main_log_folderpath = params['Logs']['Logs_Folder']
data_split_path = params['Logs']['data_split']

file_handler = logging.FileHandler(os.path.join(
    main_log_folderpath, data_split_path))
formatter = logging.Formatter(
    '%(asctime)s : %(levelname)s : %(filename)s : %(message)s')

file_handler.setFormatter(formatter)
logger.addHandler(file_handler)


class DataSplit:

    def __init__(self) -> None:
        pass

    def split_data(self):

        """This method is used to split the data into training data and testing data

        Parameters
        -----------

        None

        Returns
        --------
        None
        """

        try:
            ## Reading the data
            qsar = pd.read_csv('Data/qsar_fish_toxicity.csv', sep=';', header=None)
            qsar.columns = ['CICO', 'SM1_Dz(Z)', 'GATS1i', 'NdsCH', 'NdssC', 'MLOGP', 'LC50(mol/L)']

            ## Looking into the data
            df = qsar.copy()

            ## Splitting the data into independent features and dependent features
            X, y = df.drop('LC50(mol/L)', axis=1), df['LC50(mol/L)']

            ## Splitting the data into training data and testing data
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.05, random_state=34, stratify=X[['NdsCH']])
            
            logger.info('Input data split into X_train, X_test, y_train, y_test')
            ## Saving the train and test data into csv files
            data_folder_name = params['Data']['Data_Folder']
            train_data_name = params['Data']['Train_Data']
            test_data_name = params['Data']['Test_Data']

            logger.info('Training and testing data saved to a folder')
            pd.concat([X_train, pd.DataFrame(y_train, columns=['LC50(mol/L)'])], axis=1).to_csv(os.path.join(data_folder_name,train_data_name))
            pd.concat([X_test, pd.DataFrame(y_test, columns=['LC50(mol/L)'])], axis=1).to_csv(os.path.join(data_folder_name,test_data_name))
        
        except Exception as e:
            logger.error(e)
            raise e
        

if __name__ == "__main__":
    
    ds = DataSplit()
    ds.split_data()