import numpy as np 

VERY_SMALL_NUM = 1e-20

class Normalizer():
    def __init__(self):
        self.mean=None
        self.std=None
    
    def fit(self,tX):
        self.mean = np.mean(tX,0)
        self.std = np.std(tX,0)
        
    def transform(self,tX):  
        tX =(tX-self.mean)/(self.std+VERY_SMALL_NUM)
        return tX
    
    def fit_transform(self, tX):
        self.fit(tX)
        return self.transform(tX)

class Imputer():
    def __init__(self,missing_value=-999, dropnan=False,replacenan=None):
        assert type(missing_value)==int, "This version handles missing value in form of int only"
        self.missing_value = missing_value
        self.dropnan=dropnan
        self.replacenan=replacenan
        self.value = None
        
    def fit(self,tX):
        tX_ = tX.copy()
        self.notnan_cols = np.all(tX_!=self.missing_value,axis=0)
        if self.replacenan=='mean':
            tX_[tX_==self.missing_value]=np.nan
            self.value=np.nanmean(tX_,0)
        elif self.replacenan=='median':
            tX_[tX_==self.missing_value]=np.nan
            self.value=np.nanmedian(tX_,0)
        elif self.replacenan=='most_frequent':
            tX_[tX_==self.missing_value]=np.nan
            self.value=[]
            for i in range(tX_.shape[1]):
                tX_i = tX_[:,i]
                self.value.append(np.unique(tX_i[~np.isnan(tX_i)])[0])
            self.value=np.array(self.value)
        elif type(self.replacenan)==int:
            self.value=self.replacenan
        
    def transform(self,tX):
        if self.dropnan:
            tX=tX[:,self.notnan_cols]
        elif self.value is not None:
            inds = np.where(tX==self.missing_value)
            tX[inds] = np.take(self.value, inds[1])

        return tX
    
    def fit_transform(self, tX):
        self.fit(tX)
        return self.transform(tX)
    
class PolynomialFeature():
    def __init__(self,degree=1,cross_feat=False):
        self.degree=degree
        self.cross_feat = cross_feat
    
    def create_crossfeat(self, tX,degree):

        indices = [np.arange(tX.shape[1]) for i in range(degree)]
        combinations = np.array(np.meshgrid(*indices)).T.reshape(-1,degree)
        combinations=combinations[((combinations[:,1:]-combinations[:,:-1])>0).all(axis=1)]

        new_feats = []
        for combination in combinations:
            new_feats.append(np.prod(tX[ :,combination],axis=1,keepdims=True))

        return np.concatenate(new_feats,axis=1)
    
    def fit(self, tX):
        pass 

    def transform(self,tX):
        if self.degree >1:
            tXs = [tX]
            for d in range(2,self.degree+1):
                tXs.append(tX**d)
            if self.cross_feat:
                tXs.append(self.create_crossfeat(tX,self.degree))
            tX = np.concatenate(tXs, axis=1)
        return tX

    def fit_transform(self,tX):
        return self.transform(tX)

class PolynomialFeature():
    def __init__(self,degree=1,cross_feat=False):
        self.degree=degree
        self.cross_feat = cross_feat
    
    def create_crossfeat(self, tX,degree):

        indices = [np.arange(tX.shape[1]) for i in range(degree)]
        combinations = np.array(np.meshgrid(*indices)).T.reshape(-1,degree)
        combinations=combinations[((combinations[:,1:]-combinations[:,:-1])>0).all(axis=1)]

        new_feats = []
        for combination in combinations:
            new_feats.append(np.prod(tX[ :,combination],axis=1,keepdims=True))

        return np.concatenate(new_feats,axis=1)
    
    def transform(self,tX):
        if self.degree >1:
            tXs = [tX]
            for d in range(2,self.degree+1):
                tXs.append(tX**d)
            if self.cross_feat:
                tXs.append(self.create_crossfeat(tX,self.degree))
            tX = np.concatenate(tXs, axis=1)
        return tX

    def fit_transform(self,tX):
        return self.transform(tX)

class NonLinearTransformer():
    def __init__(self, functs):
        assert len(functs) >0, "functs must not be empty"
        self.functs = functs 

    def fit(self, tX):
        pass
    
    def transform(self,tX):
        tXs = [tX]
        for funct in self.functs:
            tXs.append(funct(tX))
        return np.concatenate(tXs, axis=-1)

    def fit_transform(self, tX):
        self.transform(tX)

class Pipeline():
    def __init__(self,*transformers):
        self.transformers = transformers

    def fit(self, tX):
        for transformer in self.transformers:
            tX = transformer.fit_transform(tX)

    def transform(self, tX, add_bias=True):
        for transformer in self.transformers:
            tX = transformer.transform(tX)
        if add_bias:
            tX = np.concatenate([tX,np.ones((tX.shape[0],1))], axis=1)
        return tX

    def fit_transform(self, tX, add_bias=True):
        for transformer in self.transformers:
            tX = transformer.fit_transform(tX)
        if add_bias:
            tX = np.concatenate([tX,np.ones((tX.shape[0],1))], axis=1)
        return tX

def remove_outliers(tX, y, k=3):
    mu, sigma = np.mean(tX, axis=0), np.std(tX, axis=0, ddof=1)
    keep_indices=np.all(np.abs((tX - mu) / (sigma+VERY_SMALL_NUM)) < k, axis=1)
    return tX[keep_indices], y[keep_indices]