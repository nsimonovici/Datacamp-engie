# Compare Algorithms
import pandas as pd
import time
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.neighbors import KNeighborsClassifier

from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.preprocessing import PolynomialFeatures, SplineTransformer
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.metrics import make_scorer, roc_auc_score, accuracy_score, f1_score

from xgboost import XGBClassifier

import mlflow
import mlflow.sklearn

class ClassificationModelComparison:
    """
    Class for regression model comparison using scikit-learn
    """

    def __init__(self, X, y, scorings=['accuracy'], test_size=0.1, seed=3, mlflow=False):
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

    def make_scorers(self, scorings):
        self.scorers = {}
        for score in scorings:
            if score == 'accuracy':
                self.scorers[score] = {}
                self.scorers[score]['name'] = 'accuracy'
                self.scorers[score]['metric'] = make_scorer(accuracy_score)
            elif score == 'f1':
                self.scorers[score] = {}
                self.scorers[score]['name'] = 'f1'
                self.scorers[score]['metric'] = make_scorer(f1_score)
            elif score == 'roc_auc':
                self.scorers[score] = {}
                self.scorers[score]['name'] = 'roc_auc'
                self.scorers[score]['metric'] = make_scorer(roc_auc_score)
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


    def run_comparison(
            self,
            preproc=['base'],
            model_param={'logistic_regression': {}},
            refit_metric='accuracy',
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
            if mdl == 'logistic_regression':
                models.append((mdl, LogisticRegression(), params))
            elif mdl == 'knn':
                models.append((mdl, KNeighborsClassifier(), params))
            elif mdl == 'xgboost':
                models.append((mdl, XGBClassifier(objective='binary:logistic', random_state=self.SEED), params))
            elif mdl == 'randomforest':
                models.append((mdl, RandomForestClassifier(), params))
            elif mdl == 'grd_boosting':
                models.append((mdl, GradientBoostingClassifier(), params))
        #   elif model == 'smf':
        #        pass
        #   elif mdl == 'xgboost':
        #        pass
            else:
                raise ValueError(f"Wrong model name. Names are to be one of : {self.allowed_models}")

        # Tout ajouter à une table pour tout sauvegarder (MLFLOW !)        
        results = []

        print("###### Start comparison ######")

        if self.USE_MLFLOW is True:
            mlflow.set_tracking_uri('http://localhost:5000')
            # Set a tag that we can use to remind ourselves what this run was for
            # mlflow.set_tag("Training Info", "Basic LR model for iris data")
            mlflow.start_run()

        start_time = time.time()

        for preproc_name, preprocessor in zip(preproc, preprocs) :
            print(f"Using preprocessor : {preproc_name}")

            for reg_name, classifier, params in models :
                print(f"Using classifier : {reg_name}")
                if verbose:
                    print(f"with specific params {params}")

                # Pipeline
                pip = Pipeline(steps=[
                    ('preproc', preprocessor),
                    ('classifier', classifier),
                ])
                
                # Grid Search
                grid = GridSearchCV(
                        estimator=pip, 
                        param_grid=params, 
                        cv=nfolds, 
                        scoring=[value['name'] for key, value in self.scorers.items()], 
                        refit=self.scorers[refit_metric]['name']
                    )
                grid.fit(self.X_train_val, self.y_train_val)
            
                if verbose:
                    print(grid.best_estimator_)
                    print(grid.best_score_)
                    print(grid.best_params_)

                best_params = grid.best_params_
                best_estimator = grid.best_estimator_

                grid_results = [reg_name] # - Nom de l'algorithme utilisé

                metrics = {}
                for scorer in self.scorers.keys():
                    metrics[scorer] = {}
                    score_train_val = self.scorers[scorer]['metric'](estimator=grid, X=self.X_train_val, y_true=self.y_train_val)
                    prevision = self.scorers[scorer]['metric'](estimator=grid, X=self.X_test, y_true=self.y_test)
                    
                    metrics[scorer]['train'] = score_train_val
                    metrics[scorer]['test'] = prevision

                    grid_results.append(prevision) # - Score pour chaque métrique sur jeu de test
                    grid_results.append(score_train_val) # - Score pour chaque métrique sur jeu de train/val ? 

                    print(f"Prevision score using ## {scorer} ## on test set = {prevision}") 
            
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

                        for metric in metrics.keys():
                            mlflow.log_metric(metric + '_test', metrics[metric]['test'])
                            mlflow.log_metric(metric + '_train', metrics[metric]['train'])

                        # Consigner le modèle
                        # mlflow.sklearn.log_model(classifier, "model")

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

        index_results = ['model_name']
        for scorer in self.scorers.keys():
            index_results.append(scorer + '_test')
            index_results.append(scorer + '_train')

        index_results += ['model', 'params']
        df_results = pd.DataFrame(results, columns=index_results)

        return df_results