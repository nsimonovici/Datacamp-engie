# Compare Algorithms
import pandas as pd
import time
from sklearn.model_selection import train_test_split, GridSearchCV, KFold
from sklearn.linear_model import LinearRegression, Ridge, LassoCV, ElasticNetCV
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor

from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.preprocessing import PolynomialFeatures, SplineTransformer
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.metrics import make_scorer, mean_absolute_error, mean_squared_error

from xgboost import XGBRegressor

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
        self.test_size = test_size # Test size in case of train/val split
        self.scorings = scorings
        self.allowed_models = ['linear_regression', 'ridge', 'lasso', 'elasticnet', 'randomforest', 'grd_boosting']
        self.allowed_scalers = ['standard', 'minmax']
        self.allowed_preproc = ['base', 'poly', 'splines']
        self.allowed_scorings = ['mae', 'mse']

        self.make_scorers(scorings)

    def split_dataset(self):
        print(f"##### Splitting Dataset with test_size = {self.test_size} and random_state = {self.SEED}")
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(self.X, self.y, test_size=self.test_size, random_state=self.SEED)

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

    def preprocessing(self, numerical_features, other_features, categorical_features=None, nknots=4, poly_order=2, scaler='standard', verbose=False):

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

        if categorical_features:
            categorical_transformer = Pipeline(steps=[
                ('imputer', SimpleImputer(strategy= 'most_frequent')),
                ('ohe', OneHotEncoder(handle_unknown= 'ignore')) # (drop='if_binary', sparse_output=False)
            ])

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
        if categorical_features:
            self.preprocessor = ColumnTransformer(transformers=[
                ('num', numeric_transformer, numerical_features + other_features),
                ('cat', categorical_transformer, categorical_features)
            ])
            self.preprocessor_ps = ColumnTransformer(transformers=[
                ('num', numeric_transformer, numerical_features),
                ('cat', categorical_transformer, categorical_features),
                ('poly', poly_transformer, other_features),
            ])
            self.preprocessor_spline = ColumnTransformer(transformers=[
                ('num', numeric_transformer, numerical_features),  # Caractéristiques numériques classiques
                ('cat', categorical_transformer, categorical_features),  # Caractéristiques catégorielles
                ('spline', spline_transformer, other_features)  # Caractéristiques pour SplineTransformer
            ])
        else:
            self.preprocessor = ColumnTransformer(transformers=[
                ('num', numeric_transformer, numerical_features + other_features),
            ])
            self.preprocessor_ps = ColumnTransformer(transformers=[
                ('num', numeric_transformer, numerical_features),
                ('poly', poly_transformer, other_features),
            ])
            self.preprocessor_spline = ColumnTransformer(transformers=[
                ('num', numeric_transformer, numerical_features),  # Caractéristiques numériques classiques
                ('spline', spline_transformer, other_features)  # Caractéristiques pour SplineTransformer
            ])

        if verbose:
            print(f"##### Preprocessors prepared : ")
            print(f"## 'base' :  {self.preprocessor}")
            print(f"## 'poly' : {self.preprocessor_ps}")
            print(f"## 'splines' : {self.preprocessor_spline}")

    def get_preproc(self, preprocessors):
        """
        Iterate over preprocessors names list and output appropriate preprocessors
        """
        preprocs = []
        for prep in preprocessors:
            if prep == 'base':
                preprocs.append(self.preprocessor)
            elif prep == 'poly':
                preprocs.append(self.preprocessor_ps)
            elif prep == 'splines':
                preprocs.append(self.preprocessor_spline)
            else:
                raise ValueError(f"Wrong preprocessor name. Names are to be one of : {self.allowed_preproc}")
            
        return preprocs

    def get_models(self, models_params):
        """
        Iterate over preprocessors names list and output appropriate preprocessors
        """
        
        models = []
        for mdl, params in models_params.items():
            if mdl == 'linear_regression':
                models.append((mdl, LinearRegression(), params))
            elif mdl == 'ridge':
                models.append((mdl, Ridge(), params))
            elif mdl == 'lasso':
                models.append((mdl, LassoCV(), params))
            elif mdl == 'elasticnet':
                models.append((mdl, ElasticNetCV(), params))
            elif mdl == 'xgboost':
                models.append((mdl, XGBRegressor(objective='reg:squarederror', random_state=self.SEED), params))
            elif mdl == 'randomforest':
                models.append((mdl, RandomForestRegressor(), params))
            elif mdl == 'grd_boosting':
                models.append((mdl, GradientBoostingRegressor(), params))
            else:
                raise ValueError(f"Wrong model name. Names are to be one of : {self.allowed_models}")
            
        return models

    def run_comparison(
            self,
            preproc=['base'],
            model_param={'linear_regression': {}},
            refit_metric='mae',
            nfolds=4, # Number of folds in case KFolds cross validation in addition to gridsearch 
            gs_nfolds=5, # Number of folds for gridsearch hyperparameter optimization
            verbose=False
            ):
        """
        Tester les différents modèles
        Pour chaque modèle + chaque preprocessing tester différentes valeurs d'hyperparamètres en grid-search
        """

        preprocs = self.get_preproc(preproc) # List of preprocessors
        models = self.get_models(model_param) # List of algorithms and parameters

        # Tout ajouter à une table pour tout sauvegarder (MLFLOW !)        
        self.results = []

        print("###### Start comparison ######")

        if self.USE_MLFLOW is True:
            mlflow.set_tracking_uri('http://localhost:5000')
            # Set a tag that we can use to remind ourselves what this run was for
            # mlflow.set_tag("Training Info", "Basic LR model for iris data")
            mlflow.start_run()

        start_time = time.time()

        for preproc_name, preprocessor in zip(preproc, preprocs) :
            print(f"Using preprocessor : {preproc_name}")

            if nfolds: # KFolds cross validation
                print(f"Using Cross-Validation")
                
                self.table_res = self.y.copy() # pandas Dataframe
                for reg_name, regressor_, params_ in models :
                    self.table_res[reg_name] = 0.0 # Initialisation de la table de prédiction optimisées

                kf = KFold(n_splits=nfolds, shuffle=True, random_state=self.SEED)
                fold = 0
                for train_index, val_index in kf.split(self.X):
                    print(f"Calculating Fold number : {fold}")
                    X_train_fold = self.X.iloc[train_index,:]
                    y_train_fold = self.y.iloc[train_index,:]
                    X_val_fold = self.X.iloc[val_index,:]
                    y_val_fold = self.y.iloc[val_index,:]

                    self.table_res = self.recursive_grid_search(
                        X_train_fold, y_train_fold, X_val_fold, y_val_fold,
                        self.table_res, val_index,
                        preproc_name, preprocessor, models,
                        nfolds=gs_nfolds, refit_metric=refit_metric,
                        fold=fold,
                        verbose=False
                        )
                    
                    fold += 1
                    
            else: # Train/Valid only                
                self.split_dataset() # Splitting dataset
                
                self.table_res = self.y_test.copy() # pandas Dataframe
                for reg_name, regressor_, params_ in models :
                    self.table_res[reg_name] = 0 # Initialisation de la table de prédiction optimisées

                ids_table = self.table_res.index

                self.table_res = self.recursive_grid_search(
                    self.X_train, self.y_train, self.X_test, self.y_test,
                    self.table_res, ids_table,
                    preproc_name, preprocessor, models,
                    nfolds=gs_nfolds, refit_metric=refit_metric,
                    fold=None,
                    verbose=False
                    )
                
        duration = time.time() - start_time
        print(f"Total duration of comparison = {duration / 60} minutes")
       
    def recursive_grid_search(self,
                    X_train, y_train, X_valid, y_valid,
                    table_results, ids_table,
                    preproc_name, preprocessor, models,
                    nfolds=5, refit_metric='mae', fold=None, verbose=False):
        """
        Loop over models using gridsearch
        """
        for reg_name, regressor, params in models :
            print(f"Using regressor : {reg_name}")
            if verbose:
                print(f"with specific params {params}")

            # Pipeline
            pip = Pipeline(steps=[
                ('preproc', preprocessor),
                ('regressor', regressor),
            ])

            if reg_name in ['lasso', 'elasticnet']:
                grid = pip.fit(X_train, y_train.values.ravel())            
            
                optimized_pred = grid.predict(X_valid)
                best_params = grid.get_params()
                best_estimator = grid # Model is fitted on all training set at the end of CV with best params           
                    
            else:
                if reg_name == 'ridge': # Using better coefficients for Ridge (from class with Eric)
                    lasso_fitted = LassoCV().fit(X_train, y_train.values.ravel())
                    path_ridge = lasso_fitted.alphas_ * 100
                    params = {"regressor__alpha": path_ridge}

                # Grid Search
                grid = GridSearchCV(
                        estimator=pip, 
                        param_grid=params, 
                        cv=nfolds, 
                        scoring=[value['name'] for key, value in self.scorers.items()], 
                        refit=self.scorers[refit_metric]['name']
                    )
                grid.fit(X_train, y_train.values.ravel())

                optimized_pred = grid.predict(X_valid)
                best_params = grid.best_params_
                best_estimator = grid.best_estimator_

            table_results.loc[ids_table, reg_name] = optimized_pred # Assign prediction to appropriate index of predictions for regressor

            self.record_best_model( # Save models, parameters and intermediate metrics
                grid,
                X_train, y_train,
                X_valid, y_valid,
                preproc_name,
                reg_name,
                best_estimator,
                best_params,
                fold=fold
                )

        return table_results # Optimized 

        #             if verbose:
        #                 print(grid.best_estimator_)
        #                 print(grid.best_score_)
        #                 print(grid.best_params_)
        #                 print(f"Refit finalized using {refit_metric}")


        #         ### ON VEUT CONSIGNER UN TABLEAU DE VALEURS PREDITES
        #         ### UNE CELLULE = PREDICTION OPTIMISEE POUR UN ALGO (PAR GRIDSEARCH) ET UN INDIVIDU

                    
    ### on va faire les métriques dans une méthode à part, on garde que la notion de refit metric et de scoring pour GS
    def record_best_model(self, grid, X_train, y_train, X_test, y_test, preproc_name, reg_name, best_estimator, best_params, fold=None):
        grid_results = [preproc_name, reg_name] # - Noms de l'algorithme et du préprocessing utilisés
        grid_results.append(fold) # Fold number if relevant

        metrics = {}
        for scorer in self.scorers.keys():
            metrics[scorer] = {}
            score_train_val = self.scorers[scorer]['metric'](estimator=grid, X=X_train, y_true=y_train)
            prevision = self.scorers[scorer]['metric'](estimator=grid, X=X_test, y_true=y_test)
            
            metrics[scorer]['train'] = score_train_val
            metrics[scorer]['test'] = prevision

            grid_results.append(prevision) # - Score pour chaque métrique sur jeu de test
            grid_results.append(score_train_val) # - Score pour chaque métrique sur jeu de train/val ? 

            print(f"Prevision score using ## {scorer} ## on test set = {prevision}") 
    
        grid_results.append(best_estimator) # - Meilleur estimateur
        grid_results.append(best_params) # - Paramètres du meilleur estimateur

        self.results.append(grid_results)

    def get_df_results(self):

        index_results = ['preproc_name', 'model_name', 'fold']
        for scorer in self.scorers.keys():
            index_results.append(scorer + '_test')
            index_results.append(scorer + '_train')

        index_results += ['model', 'params']
        df_results = pd.DataFrame(self.results, columns=index_results)

        return df_results

        
    def get_metrics(self):
        """
        Apply wanted metrics to final cross-validated and optimized table
        """

        scores = {}
        for col in self.table_res.columns[1:]:
            scores[col] = {}
            for metric in self.scorers: 
                scores[col][metric] = 0
                if metric == 'mae':
                    scores[col][metric] = mean_absolute_error(y_true=self.table_res.iloc[:, 0], y_pred=self.table_res.loc[:, col])
                elif metric == 'mse':
                    scores[col][metric] = mean_squared_error(y_true=self.table_res.iloc[:, 0], y_pred=self.table_res.loc[:, col])

        df_scores = pd.DataFrame(scores).T

        return df_scores

    def final_grid_search(self,
            X_train, y_train,
            preproc_name, model_and_params,
            nfolds=5, refit_metric='mae', verbose=False):
        """
        Final gridsearch over training set
        """
        print("Final gridsearch over training set")
        
        preprocessor = self.get_preproc(preproc_name)[0] # Preprocessor
        reg_name, regressor, params = self.get_models(model_and_params)[0] # Algorithm and parameters

        print(f"Using regressor : {reg_name}")

        # Pipeline
        pip = Pipeline(steps=[
            ('preproc', preprocessor),
            ('regressor', regressor),
        ])

        if reg_name in ['lasso', 'elasticnet']:
            grid = pip.fit(X_train, y_train.values.ravel())            
        
            optimized_estimation = grid.predict(X_train)
            best_params = grid.get_params()
            best_estimator = grid # Model is fitted on all training set at the end of CV with best params           
                
        else:
            if reg_name == 'ridge': # Using better coefficients for Ridge (from class with Eric)
                lasso_fitted = LassoCV().fit(X_train, y_train.values.ravel())
                path_ridge = lasso_fitted.alphas_ * 100
                params = {"regressor__alpha": path_ridge}

            # Grid Search
            grid = GridSearchCV(
                    estimator=pip, 
                    param_grid=params, 
                    cv=nfolds, 
                    scoring=[value['name'] for key, value in self.scorers.items()], 
                    refit=self.scorers[refit_metric]['name']
                )
            grid.fit(X_train, y_train.values.ravel())

            optimized_estimation = grid.predict(X_train)
            best_params = grid.best_params_
            best_estimator = grid.best_estimator_ # Using refit the model is fitted when outputted

        # Estimation metrics
        scores = {}        
        for metric in self.scorers: 
            scores[metric] = 0
            if metric == 'mae':
                scores[metric] = mean_absolute_error(y_true=y_train, y_pred=optimized_estimation)
            elif metric == 'mse':
                scores[metric] = mean_squared_error(y_true=y_train, y_pred=optimized_estimation)

        # Output model, parameters and estimation metrics
        self.final_best_model = scores | { # Merging dictionnaries
            'preproc_name': preproc_name,
            'reg_name': reg_name,
            'preprocessor': preprocessor,
            'best_estimator': best_estimator,
            'best_params': best_params
        }

        return self.final_best_model # Optimized

        # if self.USE_MLFLOW:
        #     with mlflow.start_run(nested=True):
        #         # Consigner le nom du modèle
        #         mlflow.set_tag("Model name", reg_name)
                
        #         # Consigner les paramètres et les métriques
        #         for param, value in best_params.items():
        #             mlflow.log_param(param, value)

        #         for metric in metrics.keys():
        #             mlflow.log_metric(metric + '_test', metrics[metric]['test'])
        #             mlflow.log_metric(metric + '_train', metrics[metric]['train'])

        #         # Consigner le modèle
        #         # mlflow.sklearn.log_model(regressor, "model")

        #         # # Infer the model signature
        #         # signature = infer_signature(X_train, lr.predict(X_train))

        #         # # Log the model
        #         # model_info = mlflow.sklearn.log_model(
        #         #     sk_model=lr,
        #         #     artifact_path="iris_model",
        #         #     signature=signature,
        #         #     input_example=X_train,
        #         #     registered_model_name="tracking-quickstart",
        #         # )

        

    # if self.USE_MLFLOW:
    #     mlflow.log_param("comparison_duration", duration)

    # if self.USE_MLFLOW:
    #     mlflow.end_run()