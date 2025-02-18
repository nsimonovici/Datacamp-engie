# Compare Algorithms
import pandas as pd
import time
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.linear_model import LinearRegression, Ridge, ElasticNet, Lasso, LassoCV, ElasticNetCV
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor

from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder # Utile ?
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.preprocessing import PolynomialFeatures, SplineTransformer
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.metrics import make_scorer, mean_absolute_error, mean_squared_error

import mlflow
import mlflow.sklearn

class RegressionModelComparison:
    """
    Class for regression model comparison using scikit-learn
    """

    def __init__(self, X, y, scorings=['mae'], test_size=0.1, seed=3, mlflow=False):
        self.SEED = seed # prepare configuration for cross validation test harness
        self.USE_MLFLOW = mlflow
        self.X = X
        self.y = y
        self.test_size = test_size
        self.scorings = scorings
        self.allowed_models = ['linear_regression', 'ridge', 'lasso', 'elasticnet', 'randomforest', 'grd_boosting']
        self.allowed_scalers = ['standard', 'minmax']
        self.allowed_preproc = ['base', 'poly', 'splines']
        self.allowed_scorings = ['mae', 'mse']

        self.make_scorers(scorings)
        self.split_dataset() # Splitting dataset

    def split_dataset(self):
        print(f"##### Splitting Dataset with test_size = {self.test_size} and random_state = {self.SEED}")
        self.X_train_val, self.X_test, self.y_train_val, self.y_test = train_test_split(self.X, self.y, test_size=self.test_size, random_state=self.SEED)
        # X_train, X_val, y_train, y_val = train_test_split(X_train_val, y_train_val, test_size=0.2, random_state=self.SEED)

    def make_scorers(self, scorings):
        self.scorers = {}
        for score in scorings:
            if score == 'mae':
                self.scorers[score] = {}
                self.scorers[score]['name'] = 'neg_mean_absolute_error'
                self.scorers[score]['metric'] = make_scorer(mean_absolute_error)
            elif score == 'mse':
                self.scorers[score] = {}
                self.scorers[score]['name'] = 'neg_mean_squared_error'
                self.scorers[score]['metric'] = make_scorer(mean_squared_error)
            else:
                raise ValueError(f"Wrong metric name. Names are to be one of : {self.allowed_scorings}")

    def preprocessing(self, numerical_features, other_features, nknots=4, poly_order=2, scaler='standard', verbose=False):

        #### Pipeline steps

        # Tester différents préprocessings avec polynomes, splines, différents scalers if necessary (depends on the algorithm)

        ## SCALER
        if scaler == 'standard':
            scaler = StandardScaler()
        elif scaler == 'minmax':
            scaler = MinMaxScaler()
        else:
            raise ValueError(f"Wrong model name. Names are to be one of : {self.allowed_scalers}")

        ## COLUMNS TRANSFORMERS
        numeric_transformer = Pipeline(steps=[
            ('imputer', SimpleImputer(strategy= 'mean')),
            ('scaler', StandardScaler())
        ])
        # categorical_transformer = Pipeline(steps=[
        #     ('imputer', SimpleImputer(strategy= 'most_frequent')),
        #     ('ohe', OneHotEncoder(handle_unknown= 'ignore')) # (drop='if_binary', sparse_output=False)
        # ])

        poly_transformer = Pipeline(steps=[
            ('imputer', SimpleImputer()),
            ('poly', PolynomialFeatures(poly_order)), #interaction_only= True
            ('scaler', StandardScaler())
        ])

        spline_transformer = Pipeline(steps=[
            ('imputer', SimpleImputer(strategy='mean')),  # Imputation des valeurs manquantes
            ('spline', SplineTransformer(degree=3, n_knots=nknots, include_bias=False)),
            ('scaler', StandardScaler())  # Mise à l'échelle des splines
        ])

        ## PROCESSORS
        self.preprocessor = ColumnTransformer(transformers= [
            ('num', numeric_transformer, numerical_features),
            # ('cat', categorical_transformer, categorical_features)
        ])

        self.preprocessor_ps = ColumnTransformer(transformers=[
            ('num', numeric_transformer, numerical_features),
            # ('cat', categorical_transformer, categorical_features),
            ('poly', poly_transformer, other_features),
        ])

        self.preprocessor_spline = ColumnTransformer(transformers=[
            ('num', numeric_transformer, numerical_features),  # Caractéristiques numériques classiques
            # ('cat', categorical_transformer, categorical_features),  # Caractéristiques catégorielles
            # ('poly', poly_transformer, other_features),  # Caractéristiques pour PolynomialFeatures
            ('spline', spline_transformer, other_features)  # Caractéristiques pour SplineTransformer
        ])

        if verbose:
            print(f"##### Preprocessors prepared : ")
            print(f"## 'base' :  {self.preprocessor}")
            print(f"## 'poly' : {self.preprocessor_ps}")
            print(f"## 'splines' : {self.preprocessor_spline}")


    def run_comparison(
            self,
            preproc=['base'],
            model_param={'linear_regression': {}},
            nfolds=5,
            verbose=False
            ):
        """
        Tester les différents modèles
        Pour chaque modèle + chaque preprocessing tester différentes valeurs d'hyperparamètres en grid-search
        """

        preprocs = []
        for prep in preproc:
            if prep == 'base':
                preprocs.append(self.preprocessor)
            elif prep == 'poly':
                preprocs.append(self.preprocessor_ps)
            elif prep == 'splines':
                preprocs.append(self.preprocessor_spline)
            else:
                raise ValueError(f"Wrong preprocessor name. Names are to be one of : {self.allowed_preproc}")

        models = []
        for mdl, params in model_param.items():
            if mdl == 'linear_regression':
                models.append((mdl, LinearRegression(), params))
            elif mdl == 'ridge':
                models.append((mdl, Ridge(), params))
            elif mdl == 'lasso':
                models.append((mdl, LassoCV(), params))
            elif mdl == 'elasticnet':
                models.append((mdl, ElasticNetCV(), params))
            elif mdl == 'randomforest':
                models.append((mdl, RandomForestRegressor(), params))
            elif mdl == 'grd_boosting':
                models.append((mdl, GradientBoostingRegressor(), params))
        #   elif model == 'smf':
        #        pass
        #   elif mdl == 'xgboost':
        #        pass
            else:
                raise ValueError(f"Wrong model name. Names are to be one of : {self.allowed_models}")

        # Tout ajouter à une table pour tout sauvegarder (MLFLOW !)

        # best_estimator_name_list = []
        # best_params_list = []
        # best_score_list = []
        # best_estimator_list =[]
        # predict_score_list = []

        # Sauvegarde des résultats :
        # - Nom de l'algorithme utilisé
        # - Score pour chaque métrique sur jeu de test
        # - Score pour chaque métrique sur jeu de train/val ?
        # - Meilleur estimateur
        # - Paramètres du meilleur estimateur
        
        results = []

        print("###### Start comparison ######")

        if self.USE_MLFLOW is True:
            mlflow.set_tracking_uri('http://localhost:5000')
            # Set a tag that we can use to remind ourselves what this run was for
            # mlflow.set_tag("Training Info", "Basic LR model for iris data")
            mlflow.start_run()

        start_time = time.time()

        for preprocessor in preprocs :
            print(f"Using preprocessor : {preprocessor}")

            for reg_name, regressor, params in models :
                print(f"Using regressor : {regressor}")
                if verbose:
                    print(f"with specific params {params}")

                # Pipeline
                pip = Pipeline(steps=[
                    ('preproc', preprocessor),
                    ('regressor', regressor),
                ])

                if reg_name in ['lasso', 'elasticnet']:
                    grid = pip.fit(self.X_train_val, self.y_train_val)            
                
                    best_params = grid.get_params()
                    best_estimator = grid

                    # Model is fitted on all training set at the end of CV with best params
                        
                else:
                    # Ce RIDGE ne fonctionne PAS
                    # if reg_name == 'ridge': # Using better coefficients for Ridge (from class with Eric)
                    #     path_ridge = LassoCV().fit(self.X_train_val, self.y_train_val).alphas_ * 100
                    #     params = {"ridge__alpha": path_ridge}

                    # Grid Search
                    grid = GridSearchCV(
                            estimator=pip, 
                            param_grid=params, 
                            cv=nfolds, 
                            scoring=[value['name'] for key, value in self.scorers.items()], 
                            refit=self.scorers['mae']['name']
                        ).fit(self.X_train_val, self.y_train_val)
                
                    if verbose:
                        print(grid.best_estimator_)
                        print(grid.best_score_)
                        print(grid.best_params_)

                    best_params = grid.best_params_
                    best_estimator = grid.best_estimator_

                grid_results = [reg_name] # - Nom de l'algorithme utilisé

                for scorer in self.scorers.keys():
                    score_train_val = self.scorers[scorer]['metric'](estimator=grid, X=self.X_train_val, y_true=self.y_train_val)
                    prevision = self.scorers[scorer]['metric'](estimator=grid, X=self.X_test, y_true=self.y_test)
                    
                    grid_results.append(prevision) # - Score pour chaque métrique sur jeu de test
                    grid_results.append(score_train_val) # - Score pour chaque métrique sur jeu de train/val ? 

                    print(f"Prevision score using ##{scorer}## on test set = {prevision}") 
            
                grid_results.append(best_estimator) # - Meilleur estimateur
                grid_results.append(best_params) # - Paramètres du meilleur estimateur

                results.append(grid_results)
                
                if self.USE_MLFLOW:
                    with mlflow.start_run(nested=True):
                        # Consigner le nom du modèle
                        mlflow.set_tag("Model name", reg_name)
                        
                        # Consigner les paramètres et les métriques
                        for param, value in best_params.items():
                            mlflow.log_param(param, value)
                        mlflow.log_metric(self.scorings[0], prevision)

                        # Consigner le modèle
                        # mlflow.sklearn.log_model(regressor, "model")

                        # # Infer the model signature
                        # signature = infer_signature(X_train, lr.predict(X_train))

                        # # Log the model
                        # model_info = mlflow.sklearn.log_model(
                        #     sk_model=lr,
                        #     artifact_path="iris_model",
                        #     signature=signature,
                        #     input_example=X_train,
                        #     registered_model_name="tracking-quickstart",
                        # )
        
        duration = time.time() - start_time

        if self.USE_MLFLOW:
            mlflow.log_param("comparison_duration", duration)

        if self.USE_MLFLOW:
            mlflow.end_run()

        print(f"Total duration of comparison = {duration / 60} minutes")

        index_results = ['model', 'test_score', 'train_val_score', 'params']
        df_results = pd.DataFrame(data=results, ).T


        return best_params_list, best_score_list, best_estimator_list, predict_score_list


# Tester l'implémentation sur un tout petit JDD ! Notamment pour MLFLOW

# Encapsuler la classe dans mlflow


    # if np.isnan(transformed_data).any():
    #     raise ValueError("Il y a des valeurs manquantes (NaN) dans les données transformées.")
    # else:
    #     print("Aucune valeur manquante (NaN) dans les données transformées.")