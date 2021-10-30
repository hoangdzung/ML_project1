import numpy as np 

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

        scores = []
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

            w, loss = self.model(y_train, x_train,**params, **kwargs)            

            y_pred = self.pred_functs(w, x_test)

            score = [acc_funct(y_test, y_pred) for acc_funct in self.acc_functs]
            scores.append(score)
        scores = np.array(scores).mean(0)

        if self.refit:
            tX = pipeline.fit_transform(tX)
            tX, y = addition_on_train(tX, y)
            w,loss = self.model(y,tX,**kwargs)
        else:
            w, loss = None, None
        return w, loss, scores

class GridSearchCV():
    def __init__ (self, model, pred_functs, acc_functs, params_grid, nfold=5, refit=True, seed=None):
        self.model = model 
        self.pred_functs = pred_functs 
        self.acc_functs = acc_functs 
        self.params_grid = tuple(params_grid.items())
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
        k_indices = self.build_k_indices(y)

        params_to_acc = {}
        for params in self.product(self.params_grid):
            crossval = CrossVal(self.model, self.pred_functs, self.acc_functs, self.nfold, refit=False, seed=self.seed)
            _, _, scores = crossval.fit(y,tX, pipeline, addition_on_train, addition_on_test)
            if verbose:
                print(params, scores)
            params_to_acc[tuple(params.items())] = scores.tolist() 

        best_params = max(params_to_acc, key=lambda x: params_to_acc[x][0])
        if self.refit:
            w,loss = self.model(y,tX, **dict(best_params),**kwargs)
        else:
            w, loss = None, None 
        return w, loss, best_params, params_to_acc  
