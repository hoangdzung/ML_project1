import numpy as np 
from tqdm import tqdm

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

def build_k_indices(y, k_fold, seed=None):
    """build k indices for k-fold."""
    num_row = y.shape[0]
    interval = int(num_row / k_fold)
    if seed is not None:
        np.random.seed(seed)
        indices = np.random.permutation(num_row)
    else:
        indices = np.arange(num_row)
    k_indices = [indices[k * interval: (k + 1) * interval]
                 for k in range(k_fold)]
    return np.array(k_indices)

class CrossVal():
    def __init__ (self, model, pred_functs, acc_functs, nfold=5, refit=True, seed=None):
        self.model = model 
        self.pred_functs = pred_functs 
        self.acc_functs = acc_functs 
        self.nfold = nfold
        self.refit = refit 
        self.seed = seed 

    def build_k_indices(self, y):
        """build k indices for k-fold."""
        num_row = y.shape[0]
        interval = int(num_row / self.nfold)
        if self.seed is not None:
            np.random.seed(self.seed)
            indices = np.random.permutation(num_row)
        else:
            indices = np.arange(num_row)
        k_indices = [indices[k * interval: (k + 1) * interval]
                    for k in range(self.nfold)]
        return np.array(k_indices)

    def fit(self, y,tX, pipeline=None,addition_on_train=None, addition_on_test=None, **kwargs):
        k_indices = self.build_k_indices(y)

        train_scores, test_scores = [], []
        for k in range(self.nfold):
            train_indices = np.concatenate([k_indices[i] for i in range(k_indices.shape[0]) if i!=k])
            test_indices = k_indices[k]
            x_train, x_test = tX[train_indices], tX[test_indices]
            y_train, y_test = y[train_indices], y[test_indices]
            if pipeline is not None:
                x_train = pipeline.fit_transform(x_train)
                if addition_on_train is not None:
                    x_train, y_train = addition_on_train(x_train,y_train)
                x_test = pipeline.transform(x_test)
                if addition_on_test is not None:
                    x_test, y_test = addition_on_test(x_test,y_test)

            w, loss = self.model(y_train, x_train, **kwargs)            

            y_test_pred = self.pred_functs(w, x_test)
            y_train_pred = self.pred_functs(w, x_train)

            test_scores.append([acc_funct(y_test, y_test_pred) for acc_funct in self.acc_functs])
            train_scores.append([acc_funct(y_train, y_train_pred) for acc_funct in self.acc_functs])

        test_scores_mean, test_scores_std = np.array(test_scores).mean(0), np.array(test_scores).std(0)
        train_scores_mean = np.array(train_scores).mean(0)

        if self.refit:
            tX = pipeline.fit_transform(tX)
            if addition_on_train is not None:
                tX, y = addition_on_train(tX, y)
            w,loss = self.model(y,tX,**kwargs)
        else:
            w, loss = None, None
        return w, loss, test_scores_mean, test_scores_std, train_scores_mean

class PartitionCrossVal(CrossVal):
    def __init__ (self, model, pred_functs, acc_functs, nfold=5, refit=True, seed=None):
        super(PartitionCrossVal, self).__init__( model, pred_functs, acc_functs, nfold, refit, seed)

    def fit(self, y,tX, pipeline=None,addition_on_train=None, addition_on_test=None, keep_cols_list=None, **kwargs):
        k_indices = self.build_k_indices(y)

        train_scores, test_scores = [], []
        for k in range(self.nfold):
            train_indices = np.concatenate([k_indices[i] for i in range(k_indices.shape[0]) if i!=k])
            test_indices = k_indices[k]
            x_train, x_test = tX[train_indices], tX[test_indices]
            y_train, y_test = y[train_indices], y[test_indices]
            y_test_list, y_train_list = [], []
            y_test_pred_list, y_train_pred_list = [], []
            for jet_num in range(4):
                sub_x_train = x_train[x_train[:,COL2ID['PRI_jet_num']]==jet_num]
                if keep_cols_list is not None:
                    keep_cols = np.array([COL2ID[col_name] for col_name in keep_cols_list[jet_num]])
                else:
                    keep_cols = np.array([i for i in range(sub_x_train.shape[-1]) if len(set(sub_x_train[:,i]))>1])
                sub_x_train = sub_x_train[:,keep_cols]
                sub_y_train = y_train[x_train[:,COL2ID['PRI_jet_num']]==jet_num]
                sub_x_test = x_test[x_test[:,COL2ID['PRI_jet_num']]==jet_num]
                sub_x_test = sub_x_test[:,keep_cols]
                sub_y_test = y_test[x_test[:,COL2ID['PRI_jet_num']]==jet_num]

                if pipeline is not None:
                    sub_x_train = pipeline.fit_transform(sub_x_train)
                    if addition_on_train is not None:
                        sub_x_train, sub_y_train = addition_on_train(sub_x_train,sub_y_train)
                    sub_x_test = pipeline.transform(sub_x_test)
                    if addition_on_test is not None:
                        sub_x_test, sub_y_test = addition_on_test(sub_x_test,sub_y_test)
                w, loss = self.model(sub_y_train, sub_x_train, **kwargs)            
                sub_y_test_pred = self.pred_functs(w, sub_x_test)
                sub_y_train_pred = self.pred_functs(w, sub_x_train)

                y_test_list.append(sub_y_test)
                y_test_pred_list.append(sub_y_test_pred)
                y_train_list.append(sub_y_train)
                y_train_pred_list.append(sub_y_train_pred)

            y_test = np.concatenate(y_test_list)
            y_test_pred = np.concatenate(y_test_pred_list)            
            y_train = np.concatenate(y_train_list)
            y_train_pred = np.concatenate(y_train_pred_list)

            test_scores.append([acc_funct(y_test, y_test_pred) for acc_funct in self.acc_functs])
            train_scores.append([acc_funct(y_train, y_train_pred) for acc_funct in self.acc_functs])

        test_scores_mean, test_scores_std = np.array(test_scores).mean(0), np.array(test_scores).std(0)
        train_scores_mean = np.array(train_scores).mean(0)


        if self.refit:
            print("Not support refit at the moment")
            w, loss = None, None
        else:
            w, loss = None, None
        return w, loss, test_scores_mean, test_scores_std, train_scores_mean

class MultiPartitionCrossVal(CrossVal):
    def __init__ (self, model, pred_functs, acc_functs, nfold=5, refit=True, seed=None):
        super(MultiPartitionCrossVal, self).__init__( model, pred_functs, acc_functs, nfold, refit, seed)

    def fit(self, y,tX, pipeline=None,addition_on_train=None, addition_on_test=None, **kwargs):
        k_indices = self.build_k_indices(y)

        train_scores, test_scores = [], []
        for k in range(self.nfold):
            train_indices = np.concatenate([k_indices[i] for i in range(k_indices.shape[0]) if i!=k])
            test_indices = k_indices[k]
            x_train, x_test = tX[train_indices], tX[test_indices]
            y_train, y_test = y[train_indices], y[test_indices]
            y_test_list, y_train_list = [], []
            y_test_pred_list, y_train_pred_list = [], []
            for jet_num in range(4):
                sub_x_train = x_train[x_train[:,COL2ID['PRI_jet_num']]==jet_num]
                keep_cols = np.array([i for i in range(sub_x_train.shape[-1]) if len(set(sub_x_train[:,i]))>1])
                sub_x_train = sub_x_train[:,keep_cols]
                sub_y_train = y_train[x_train[:,COL2ID['PRI_jet_num']]==jet_num]
                sub_x_test = x_test[x_test[:,COL2ID['PRI_jet_num']]==jet_num]
                sub_x_test = sub_x_test[:,keep_cols]
                sub_y_test = y_test[x_test[:,COL2ID['PRI_jet_num']]==jet_num]
                having_mass_train_indices = sub_x_train[:, COL2ID['DER_mass_MMC']]!=-999
                having_mass_test_indices = sub_x_test[:, COL2ID['DER_mass_MMC']]!=-999
                for mass_train_indices, mass_test_indices \
                    in zip([having_mass_train_indices,~having_mass_train_indices],\
                            [having_mass_test_indices, ~having_mass_test_indices]):
                    sub_x_train_mass = sub_x_train[mass_train_indices]
                    sub_y_train_mass = sub_y_train[mass_train_indices]
                    sub_x_test_mass = sub_x_test[mass_test_indices]
                    sub_y_test_mass = sub_y_test[mass_test_indices]
                    if pipeline is not None:
                        sub_x_train_mass = pipeline.fit_transform(sub_x_train_mass)
                        if addition_on_train is not None:
                            sub_x_train_mass, sub_y_train_mass = addition_on_train(sub_x_train_mass, sub_y_train_mass)
                        sub_x_test_mass = pipeline.transform(sub_x_test_mass)
                        if addition_on_test is not None:
                            sub_x_test_mass, sub_y_test_mass = addition_on_test(sub_x_test_mass,sub_y_test_mass)

                    w, loss = self.model(sub_y_train_mass, sub_x_train_mass, **kwargs)            
                    sub_y_test_pred = self.pred_functs(w, sub_x_test_mass)
                    sub_y_train_pred = self.pred_functs(w, sub_x_train_mass)

                    y_test_list.append(sub_y_test_mass)
                    y_test_pred_list.append(sub_y_test_pred)
                    y_train_list.append(sub_y_train_mass)
                    y_train_pred_list.append(sub_y_train_pred)

            y_test = np.concatenate(y_test_list)
            y_test_pred = np.concatenate(y_test_pred_list)            
            y_train = np.concatenate(y_train_list)
            y_train_pred = np.concatenate(y_train_pred_list)
            
            test_scores.append([acc_funct(y_test, y_test_pred) for acc_funct in self.acc_functs])
            train_scores.append([acc_funct(y_train, y_train_pred) for acc_funct in self.acc_functs])


        test_scores_mean, test_scores_std = np.array(test_scores).mean(0), np.array(test_scores).std(0)
        train_scores_mean = np.array(train_scores).mean(0)

        if self.refit:
            print("Not support refit at the moment")
            w, loss = None, None
        else:
            w, loss = None, None
        return w, loss, test_scores_mean, test_scores_std, train_scores_mean

class GridSearchCV():
    def __init__ (self, model, pred_functs, acc_functs, params_grid, cross_val, nfold=5, refit=True, seed=None):
        self.model = model 
        self.pred_functs = pred_functs 
        self.acc_functs = acc_functs 
        self.params_grid = tuple(params_grid.items())
        self.cross_val = cross_val
        self.nfold = nfold
        self.refit = refit 
        self.seed = seed 

    def product(self, params_grid):
        if not params_grid:
            yield {}
        else:
            key, vals = params_grid[0]
            for val in vals:
                for prod in self.product(params_grid[1:]):
                    yield {**{key:val},**prod}

    def fit(self, y,tX, pipeline=None,addition_on_train=None, addition_on_test=None, verbose=True, **kwargs):
        params_to_acc = {}
        for params in self.product(self.params_grid):
            crossval = self.cross_val(self.model, self.pred_functs, self.acc_functs, self.nfold, refit=False, seed=self.seed)
            _, _, test_scores_mean, test_scores_std, train_scores_mean = crossval.fit(y,tX, pipeline, addition_on_train, addition_on_test,**params)
            if verbose:
                print(params, test_scores_mean, train_scores_mean)
            params_to_acc[tuple(params.items())] = (test_scores_mean.tolist(), test_scores_std.tolist())

        best_params = max(params_to_acc, key=lambda x: params_to_acc[x][0][0])
        if self.refit:
            w,loss = self.model(y,tX, **dict(best_params),**kwargs)
        else:
            w, loss = None, None 
        return w, loss, params_to_acc[best_params], best_params, params_to_acc 
