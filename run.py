import numpy as np

from crossval import GridSearchCV, CrossVal, PartitionCrossVal, MultiPartitionCrossVal
from implementations import logistic_regression, reg_logistic_regression, least_squares
from proj1_helpers import load_csv_data, predict_labels, acc_score, f1_score,create_csv_submission
from preprocessing import NonLinearTransformer, Normalizer, Imputer, PolynomialFeature, Pipeline, remove_outliers

COL2ID={'DER_mass_MMC': 0,
 'DER_mass_transverse_met_lep': 1,
 'DER_mass_vis': 2,
 'DER_pt_h': 3,
 'DER_deltaeta_jet_jet': 4,
 'DER_mass_jet_jet': 5,
 'DER_prodeta_jet_jet': 6,
 'DER_deltar_tau_lep': 7,
 'DER_pt_tot': 8,
 'DER_sum_pt': 9,
 'DER_pt_ratio_lep_tau': 10,
 'DER_met_phi_centrality': 11,
 'DER_lep_eta_centrality': 12,
 'PRI_tau_pt': 13,
 'PRI_tau_eta': 14,
 'PRI_tau_phi': 15,
 'PRI_lep_pt': 16,
 'PRI_lep_eta': 17,
 'PRI_lep_phi': 18,
 'PRI_met': 19,
 'PRI_met_phi': 20,
 'PRI_met_sumet': 21,
 'PRI_jet_num': 22,
 'PRI_jet_leading_pt': 23,
 'PRI_jet_leading_eta': 24,
 'PRI_jet_leading_phi': 25,
 'PRI_jet_subleading_pt': 26,
 'PRI_jet_subleading_eta': 27,
 'PRI_jet_subleading_phi': 28,
 'PRI_jet_all_pt': 29}


DATA_TRAIN_PATH = './data/train.csv' # TODO: download train data and supply path here 
y, tX, ids = load_csv_data(DATA_TRAIN_PATH)
DATA_TEST_PATH = './data/test.csv' # TODO: download train data and supply path here 
_, tX_test, ids_test = load_csv_data(DATA_TEST_PATH)

pipeline = Pipeline(Imputer(replacenan='mean'), PolynomialFeature(degree=3), Normalizer())

y_pred = []
indices = []
for jet_num in range(4):
    sub_x_train = tX[tX[:,COL2ID['PRI_jet_num']]==jet_num]
    sub_y_train = y[tX[:,COL2ID['PRI_jet_num']]==jet_num]
    sub_x_test = tX_test[tX_test[:,COL2ID['PRI_jet_num']]==jet_num]
    ids = ids_test[tX_test[:,COL2ID['PRI_jet_num']]==jet_num]
    having_mass_train_indices = sub_x_train[:, COL2ID['DER_mass_MMC']]!=-999
    having_mass_test_indices = sub_x_test[:, COL2ID['DER_mass_MMC']]!=-999
    for mass_train_indices, mass_test_indices in zip([having_mass_train_indices,~having_mass_train_indices],                [having_mass_test_indices, ~having_mass_test_indices]):
        sub_x_train_mass = sub_x_train[mass_train_indices]
        sub_y_train_mass = sub_y_train[mass_train_indices]
        sub_x_test_mass = sub_x_test[mass_test_indices]
        sub_ids_mass = ids[mass_test_indices]
        keep_cols = np.array([i for i in range(sub_x_train_mass.shape[-1]) if len(set(sub_x_train_mass[:,i]))>1])
        sub_x_train_mass = sub_x_train_mass[:,keep_cols]
        sub_x_test_mass = sub_x_test_mass[:,keep_cols]

        sub_x_train_mass = pipeline.fit_transform(sub_x_train_mass)
        sub_x_train_mass, sub_y_train_mass = remove_outliers(sub_x_train_mass, sub_y_train_mass)
        sub_x_test_mass = pipeline.transform(sub_x_test_mass)

        w, loss = logistic_regression(sub_y_train_mass, sub_x_train_mass, gamma=1, early_stopping=True)            
        sub_y_test_pred = predict_labels(w, sub_x_test_mass)
        
        indices.append(sub_ids_mass)
        y_pred.append(sub_y_test_pred)

indices=np.concatenate(indices)
y_pred=np.concatenate(y_pred)


OUTPUT_PATH = 'results/submission.csv' # TODO: fill in desired name of output file for submission
create_csv_submission(indices, y_pred, OUTPUT_PATH)

