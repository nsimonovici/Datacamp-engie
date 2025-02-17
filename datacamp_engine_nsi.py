from regression_model_comparison import RegressionModelComparison
import numpy as np
import pandas as pd

dataX = pd.read_csv('data/engie_X.csv', header=0, sep=';', decimal='.')
dataY = pd.read_csv('data/engie_Y.csv', header=0,  sep=';', decimal='.')
data_raw = pd.merge(dataX, dataY, on='ID', how='inner')
print("SHAPE = ", data_raw.shape)

sample_size = 100
X = data_raw.drop(columns=['ID', 'MAC_CODE']).iloc[:sample_size, :]
Y = data_raw.TARGET.iloc[:sample_size]

comparison = RegressionModelComparison(
    X,
    Y,
    scorings=['mae', 'mse'],
    test_size=0.1,
    seed=3,
    mlflow=True
    )

numerical_features = [
    'Date_time', 'Pitch_angle', 'Pitch_angle_min',
       'Pitch_angle_max', 'Pitch_angle_std', 'Hub_temperature',
       'Hub_temperature_min', 'Hub_temperature_max', 'Hub_temperature_std',
       'Generator_converter_speed', 'Generator_converter_speed_min',
       'Generator_converter_speed_max', 'Generator_converter_speed_std',
       'Generator_speed', 'Generator_speed_min', 'Generator_speed_max',
       'Generator_speed_std', 'Generator_bearing_1_temperature',
       'Generator_bearing_1_temperature_min',
       'Generator_bearing_1_temperature_max',
       'Generator_bearing_1_temperature_std',
       'Generator_bearing_2_temperature',
       'Generator_bearing_2_temperature_min',
       'Generator_bearing_2_temperature_max',
       'Generator_bearing_2_temperature_std', 'Generator_stator_temperature',
       'Generator_stator_temperature_min', 'Generator_stator_temperature_max',
       'Generator_stator_temperature_std', 'Gearbox_bearing_1_temperature',
       'Gearbox_bearing_1_temperature_min',
       'Gearbox_bearing_1_temperature_max',
       'Gearbox_bearing_1_temperature_std', 'Gearbox_bearing_2_temperature',
       'Gearbox_bearing_2_temperature_min',
       'Gearbox_bearing_2_temperature_max',
       'Gearbox_bearing_2_temperature_std', 'Gearbox_inlet_temperature',
       'Gearbox_inlet_temperature_min', 'Gearbox_inlet_temperature_max',
       'Gearbox_inlet_temperature_std', 'Gearbox_oil_sump_temperature',
       'Gearbox_oil_sump_temperature_min', 'Gearbox_oil_sump_temperature_max',
       'Gearbox_oil_sump_temperature_std', 'Nacelle_angle',
       'Nacelle_angle_min', 'Nacelle_angle_max', 'Nacelle_angle_std',
       'Nacelle_temperature', 'Nacelle_temperature_min',
       'Nacelle_temperature_max', 'Nacelle_temperature_std',
       'Absolute_wind_direction', 'Outdoor_temperature',
       'Outdoor_temperature_min', 'Outdoor_temperature_max',
       'Outdoor_temperature_std', 'Grid_frequency', 'Grid_frequency_min',
       'Grid_frequency_max', 'Grid_frequency_std', 'Grid_voltage',
       'Grid_voltage_min', 'Grid_voltage_max', 'Grid_voltage_std',
       'Rotor_speed', 'Rotor_speed_min', 'Rotor_speed_max', 'Rotor_speed_std',
       'Rotor_bearing_temperature', 'Rotor_bearing_temperature_min',
       'Rotor_bearing_temperature_max', 'Rotor_bearing_temperature_std'
       ]

other_features = ['Absolute_wind_direction_c', 'Nacelle_angle_c']

comparison.preprocessing(
    numerical_features,
    other_features,
    nknots=4,
    poly_order=2,
    scaler='standard',
    )

best_params_list, best_score_list, best_estimator_list, predict_score_list = comparison.run_comparison(
                    preproc=['base'],
                    model_param={
                        'linear_regression': {},
                        # 'ridge': {},
                        'lasso': {},
                        'elasticnet': {},
                        'randomforest': {
                            'regressor__n_estimators' : [100, 200, 300, 1000], # Nombre d'arbres dans la forêt. defaut 100
                            'regressor__max_depth' : [None, 10, 20, 30], # Profondeur maximale des arbres. Si None, les arbres sont développés jusqu'à ce que toutes les feuilles soient pures ou que chaque feuille contienne moins que min_samples_split échantillons
                            'regressor__max_features': [3, 'sqrt'], # Nombre maximum de caractéristiques considérées pour chaque split (division d'un nœud en deux sous-nœuds)
                            },
                        'grd_boosting': {
                            'regressor__learning_rate' : [.00, .01, .1, 1],
                            'regressor__max_depth' : [3, 7, 9],
                            'regressor__subsample' : [0.5, 0.7, 1],
                            'regressor__n_estimators' : [100, 200, 300, 1000]
                            }
                        },
                    nfolds=5,
                    verbose=True
                    )