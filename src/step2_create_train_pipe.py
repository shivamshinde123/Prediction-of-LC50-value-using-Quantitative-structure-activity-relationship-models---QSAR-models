from step0_utility_functions import Utility
import logging
import os
import json
import joblib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.svm import SVR
from sklearn.neighbors import KNeighborsRegressor
from sklearn.ensemble import VotingRegressor
from sklearn.pipeline import Pipeline
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

Utility().create_folder('Logs')
params = Utility().read_params()

main_log_folderpath = params['Logs']['Logs_Folder']
model_training_path = params['Logs']['Model_Training']

file_handler = logging.FileHandler(os.path.join(
    main_log_folderpath, model_training_path))
formatter = logging.Formatter(
    '%(asctime)s : %(levelname)s : %(filename)s : %(message)s')

file_handler.setFormatter(formatter)
logger.addHandler(file_handler)


class ModelTraining:

    def __init__(self) -> None:
        pass

    def plot_feat_importances(self, feat_imp_df):
        """This method is used to create a plot between precision and recall.

        Parameters
        -----------

        precisions: precisions
        recalls: recalls

        Returns
        --------
        None
        """
        try:
            plt.figure(figsize=(12, 8))
            sns.set_style('darkgrid')
            plt.plot(feat_imp_df['features'], feat_imp_df['Importance'], marker='*', mec='r', color='k')
            plt.title('Feature Importance')
            plt.xlabel("Features")
            plt.ylabel("Importance")

            plots_folder = params['Plots']['Plot_Foler']
            plot_name = params['Plots']['feat_importance_plotname']

            Utility().create_folder(plots_folder)
            plt.savefig(os.path.join(plots_folder, plot_name))

        except Exception as e:
            raise e

    def find_feature_importance(self):
        """This method is used to find the importance of input data features


        Parameters
        -----------

        None

        Returns
        --------
        None
        """
        try:
            ## Loading the training and testing data
            data_folder_name = params['Data']['Data_Folder']
            train_data_name = params['Data']['Train_Data']
            feat_importance_df = params['Data']['feat_importance']

            train_data = pd.read_csv(os.path.join(data_folder_name,train_data_name))

            X_train, y_train = train_data.drop('LC50(mol/L)', axis=1), train_data['LC50(mol/L)']

            logger.info('Training data is loaded from the Data folder')

            ## Creating a object for random forest regressor
            rfr = RandomForestRegressor(bootstrap=True, oob_score=True, random_state=78)  

            logger.info('Random forest regressor model created to find out feature importance')

            ## Training the voting regressor using training data
            rfr_model = rfr.fit(X_train, y_train)
            logger.info('random forest regressor trained on training data')

            ## Feature importances
            feat_importance = pd.DataFrame()
            feat_importance['features'] = X_train.columns
            feat_importance['Importance'] = np.round(rfr_model.feature_importances_*100, 2)

            feat_importance = feat_importance.sort_values(by='Importance', ascending=False)
            logger.info('Feature importance dataframe created')

            self.plot_feat_importances(feat_importance)
            logger.info('Feature importance plot created and saved')

            feat_importance.to_csv(os.path.join(data_folder_name,feat_importance_df))
            logger.info('feat_importance dataframe saved as csv file')

        except Exception as e:
            logger.error(e)
            raise e



    def model_training(self):

        """This method is used to train a model on training data and also to evaluate the trained model on the test data


        Parameters
        -----------

        None

        Returns
        --------
        None
        """
        try:
            ## Loading the training and testing data
            data_folder_name = params['Data']['Data_Folder']
            train_data_name = params['Data']['Train_Data']
            test_data_name = params['Data']['Test_Data']

            train_data = pd.read_csv(os.path.join(data_folder_name,train_data_name))
            test_data = pd.read_csv(os.path.join(data_folder_name,test_data_name))

            X_train, y_train = train_data.drop('LC50(mol/L)', axis=1), train_data['LC50(mol/L)']
            X_test, y_test = test_data.drop('LC50(mol/L)', axis=1), test_data['LC50(mol/L)']

            logger.info('Training and testing data is loaded again from the Data folder')

            ## Creating a voting regressor model
            ## model 1: Support Vector Regression
            svr = SVR()

            ## model 2: Nearest Neighbors Regression
            knr = KNeighborsRegressor()

            ## model 3: Random Forest Regression
            rfr = RandomForestRegressor()   

            ## model 4: Gradient Boosting Regressor
            gbr = GradientBoostingRegressor()

            ## voting regressor
            vr = VotingRegressor(estimators = [('svr', svr), ('knr', knr), ('rfr', rfr), ('gbr', gbr)])
            logger.info('Voting regressor object created')

            ## creating a pipeline for preprocessing and training the voting regressor 
            ## Creating a pipeline for data preprocessing
            preprocess_pipe = Pipeline(steps=[
                ('scaling', StandardScaler())
            ])
            logger.info('Preprocessing pipeline created.')
            ## vr pipeline
            vr_pipe = Pipeline(steps=[
                ('preprocess_pipe', preprocess_pipe),
                ('vr_model', vr)
            ])
            logger.info('Final pipeline containing preprocessing and model training step created.')

            ## Training the voting regressor using training data
            vr = vr_pipe.fit(X_train, y_train)
            logger.info('voting regressor trained on training data.')

            ## Checking the performance of model on test data
            vr_predictions = vr.predict(X_test)
            logger.info('predictions were made using trained voting regressor and the test data')

            r2 = r2_score(y_test, vr_predictions)
            msr = mean_squared_error(y_test, vr_predictions)
            mae = mean_absolute_error(y_test, vr_predictions)

            logger.info('metrics r2 score, mean squared error and mean absolute error were calculated using trained voting regressor and test data')

            ## Saving the trained model metrics into json file
            metrics_folder_name = params['Model']['Metrics']['Metrics_Folder']
            metrics_file_name = params['Model']['Metrics']['Metrics_File']

            Utility().create_folder(metrics_folder_name)
            with open(os.path.join(metrics_folder_name, metrics_file_name), 'w') as json_file:
                    metrics = dict()
                    metrics['r2_score'] = r2
                    metrics['mean_squared_error'] = msr
                    metrics['mean_absolute_error'] = mae

                    json.dump(metrics, json_file, indent=4)

            logger.info('trained model metrics saved into json file')
            
            
            ## Saving the train model into joblib file
            model_folder = params['Model']['Model_Folder']
            model_name = params['Model']['Model_Name']

            Utility().create_folder(model_folder)

            model_dir = os.path.join(model_folder, model_name)

            joblib.dump(vr, model_dir)

            logger.info('Trained model saved as a joblib file.')
        
        except Exception as e:
            logger.error(e)
            raise e
        

if __name__ == "__main__":

    mt = ModelTraining()
    mt.find_feature_importance()
    mt.model_training()