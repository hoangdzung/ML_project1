import numpy as np 

VERY_SMALL_NUM = 1e-20

class Normalizer():
    """
    Standardize features by removing the mean and scaling to unit variance.
    The standard score of a sample `x` is calculated as:
        z = (x - u) / (s+very_small_value)
    where `u` is the mean of the training samples, and `s` is the standard 
    deviation of the training samples. A very small number is added to avoid
    division by zero.
    
    Parameters
    ----------
    
    Attributes
    ----------
    mean : ndarray of shape (n_features,) or None
        The mean value for each feature in the training set.
    std : ndarray of shape (n_features,) or None
        The variance for each feature in the training set. 

    Examples
    --------
    >>> from preprocessing import Normalizer
    >>> import numpy as np
    >>> scaler = Normalizer()
    >>> data = np.array([[0, 0], [0, 0], [1, 1], [1, 1]])
    >>> scaler.fit_transform(data)
    array([[-1., -1.],
        [-1., -1.],
        [ 1.,  1.],
        [ 1.,  1.]])

    """
    def __init__(self):
        self.mean=None
        self.std=None
    
    def fit(self,tX):
        """
        Compute the mean and std to be used for later scaling.
        
        Parameters
        ----------
        tX : numpy.ndarray
            The features matrix used to compute the mean and standard deviation
            used for later scaling along the features axis.
        
        Returns
        -------
        None
        """
        self.mean = np.mean(tX,0)
        self.std = np.std(tX,0)
        
    def transform(self,tX):  
        """
        Perform standardization by centering and scaling.
        
        Parameters
        ----------
        tX : numpy.ndarray
            Matrix of features of size (NxD)
        
        Returns
        -------
        numpy.ndarray
            Transformed matrix.
        """
        tX =(tX-self.mean)/(self.std+VERY_SMALL_NUM)
        return tX
    
    def fit_transform(self, tX):
        self.fit(tX)
        return self.transform(tX)

class Imputer():
    """
    Imputation transformer for completing missing values.
    
    Parameters
    ----------

    missing_value : int, float
        The value will be considered as the missing value, default -999
    dropnan : bool
        Whether to drop every feautres containing missing values, default False
    replacenan: str, int, float, None 
        Strategy to impute missing value, defautl None
        - If "mean", then replace missing values using the mean of each feature.
        - If "median", then replace missing values using the median of each feature.
        - If "most_frequent", then replace missing using the most frequent value of each feature
        - If type is int or float, then replace missing using this value
        - If None, then not replace missing values.    

    Attributes
    ----------

    missing_value, dropnan, replacenan : 
        Similar to parameters    
    value: 1-d numpy.ndarray
        Array storing imputing value for each features

    Examples
    --------
    >>> import numpy as np
    >>> from preprocessing import Imputer
    >>> data=np.array([[1,1,-999],[-999,0,0]])
    >>> imputer = Imputer(replacenan='mean')
    >>> imputer.fit_transform(data)
    array([[1., 1., 0.],
        [1., 0., 0.]])

    """

    def __init__(self,missing_value=-999, dropnan=False,replacenan=None):
        assert type(missing_value) in [int, float], "This version handles missing value in form of int or float only"
        self.missing_value = missing_value
        self.dropnan=dropnan
        self.replacenan=replacenan
        self.value = None
        
    def fit(self,tX):
        """
        Compute the replacing value for each features or list of dropped feautres.
        
        Parameters
        ----------
        tX : numpy.ndarray
            The training feature matrix.
        
        Returns
        -------
        None
        """
        tX_ = tX.astype(float).copy()
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
        
        elif type(self.replacenan) is [int, float]:
            self.value=self.replacenan
        
    def transform(self,tX):
        """
        Impute missing values or drop columns from the matrix.
        
        Parameters
        ----------
        tX : numpy.ndarray
            Matrix of features of size (NxD)
        
        Returns
        -------
        numpy.ndarray
            Transformed matrix.
        None
        """
        
        tX=tX.astype(float)

        if self.dropnan:
            tX=tX[:,self.notnan_cols]
        
        if self.value is not None:
            inds = np.where(tX==self.missing_value)
            tX[inds] = np.take(self.value, inds[1])

        return tX
    
    def fit_transform(self, tX):
        self.fit(tX)
        return self.transform(tX)
    
class PolynomialFeature():
    """
    Generate polynomial and interaction features.

    Parameters
    ----------

    degree : int
        The maximal degree of the polynomial features, default 1, mean that applying nothing at all
    cross_feat : bool
        Whether to use interactions between features, default False
        In each interaction, there are exactly `degree` features, each features occurs only once

    Attributes
    ----------

    degree, cross_feat:
        Similar to parameters
        
    Examples
    --------
    >>> import numpy as np
    >>> from preprocessing import PolynomialFeature
    >>> data = np.array([[1,2,3],[4,5,6]])
    >>> PolynomialFeature(degree=2).fit_transform(data)
    array([[ 1,  2,  3,  1,  4,  9],
        [ 4,  5,  6, 16, 25, 36]])
    >>> PolynomialFeature(degree=2,cross_feat=True).fit_transform(data)
    array([[ 1,  2,  3,  1,  4,  9,  2,  3,  6],
        [ 4,  5,  6, 16, 25, 36, 20, 24, 30]])

    """
    def __init__(self,degree=1,cross_feat=False):
        self.degree=degree
        self.cross_feat = cross_feat
    
    def create_crossfeat(self, tX, degree):
        """
        Create interaction features:
        
        Parameters
        ----------
        tX : numpy.ndarray
            Matrix of features of size (NxD)
        degree: int
            Number of features for each interaction

        Returns
        -------
        numpy.ndarray
            Transformed features.
        None
        """

        indices = [np.arange(tX.shape[1]) for i in range(degree)]
        combinations = np.array(np.meshgrid(*indices)).T.reshape(-1,degree)
        combinations=combinations[((combinations[:,1:]-combinations[:,:-1])>0).all(axis=1)]

        new_feats = []
        for combination in combinations:
            new_feats.append(np.prod(tX[ :,combination],axis=1,keepdims=True))

        return np.concatenate(new_feats,axis=1)
    
    def fit(self, tX):
        """
        This class doesn't need to learn the transformation
        """
        pass 

    def transform(self,tX):
        """
        Transform the feature matrix.
        
        Parameters
        ----------
        tX : numpy.ndarray
            Matrix of features of size (NxD)
        
        Returns
        -------
        numpy.ndarray
            Transformed matrix.
        None
        """
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
    """
    Generate augmented features using arbitrary functions.

    Parameters
    ----------

    functs : list of functions
        List of transformation functions will be applied to every features
        You have to take responsibility to check the validity of functions

    Attributes
    ----------
    functs:
        Similar to parameters
        
    Examples
    --------
    >>> import numpy as np
    >>> from preprocessing import NonLinearTransformer
    >>> data = np.array([[1,0],[3,4]])
    >>> NonLinearTransformer([np.exp, lambda x: x**2+1]).fit_transform(data)
    array([[ 1.        ,  0.        ,  2.71828183,  1.        ,  2.        ,
            1.        ],
        [ 3.        ,  4.        , 20.08553692, 54.59815003, 10.        ,
            17.        ]])

    """

    def __init__(self, functs):
        assert len(functs) >0, "functs must not be empty"
        self.functs = functs 

    def fit(self, tX):
        """
        This class doesn't need to learn the transformation
        """
        pass
    
    def transform(self,tX):
        """
        Transform the feature matrix. The augmented features 
        will be concatenated to the original features
        
        Parameters
        ----------
        tX : numpy.ndarray
            Matrix of features of size (NxD)
        
        Returns
        -------
        numpy.ndarray
            Transformed matrix.
        None
        """
        tXs = [tX]
        for funct in self.functs:
            tXs.append(funct(tX))
        return np.concatenate(tXs, axis=-1)

    def fit_transform(self, tX):
        return self.transform(tX)

class Pipeline():
    """
    Pipeline of data transforms, apply on feature matrix

    Parameters
    ----------

    transformers : list of transformation objects
        List of transformation functions will be applied to features matrix
        that are chained, in the order in which they are chained

    Attributes
    ----------
    transformers:
        Similar to parameters
        
    Examples
    --------
    >>> from preprocessing import Imputer, Normalizer, Pipeline
    >>> pipeline = Pipeline(Imputer(missing_value=0, dropnan=True), Normalizer())
    >>> data=np.array([[0,1,2],[3,4,5]])
    >>> pipeline.fit_transform(data)
    array([[-1., -1.,  1.],
        [ 1.,  1.,  1.]])
    >>> pipeline.fit_transform(data,add_bias=False)
    array([[-1., -1.],
        [ 1.,  1.]])

    """   

    def __init__(self,*transformers):
        self.transformers = transformers

    def fit(self, tX):
        """
        Fit the feature matrix to learn tranformation functions
        
        Parameters
        ----------
        tX : numpy.ndarray
            Matrix of features of size (NxD)
        
        Returns
        -------
        None
        """
        for transformer in self.transformers:
            tX = transformer.fit_transform(tX)

    def transform(self, tX, add_bias=True):
        """
        Transform the feature matrix.
        
        Parameters
        ----------
        tX : numpy.ndarray
            Matrix of features of size (NxD)
        add_bias:bool
            Whether to add the bias term
        Returns
        -------
        numpy.ndarray
            Transformed matrix.
        """

        for transformer in self.transformers:
            tX = transformer.transform(tX)
        if add_bias:
            tX = np.concatenate([tX,np.ones((tX.shape[0],1))], axis=1)
        return tX

    def fit_transform(self, tX, add_bias=True):
        """
        Fit and then ransform the feature matrix.
        
        Parameters
        ----------
        tX : numpy.ndarray
            Matrix of features of size (NxD)
        add_bias:bool
            Whether to add the bias term
        Returns
        -------
        numpy.ndarray
            Transformed matrix.
        """

        for transformer in self.transformers:
            tX = transformer.fit_transform(tX)
        if add_bias:
            tX = np.concatenate([tX,np.ones((tX.shape[0],1))], axis=1)
        return tX

def remove_outliers(tX, y, k=3):
    """
    Remove samples which have at least one outlier feature value.
    A feature value is considered as an outlier if its z_score larger than or equal to k
    
    Parameters
    ----------
    tX : numpy.ndarray
        Matrix of features of size (NxD)
    y:numpy.ndarray
        Vector of labels of size N
    k: float, int
        Threshold of z_score
    Returns
    -------
    (numpy.ndarray, numpy.ndarray)
        Filted features and labels.
    """
    mu, sigma = np.mean(tX, axis=0), np.std(tX, axis=0, ddof=1)
    keep_indices=np.all(np.abs((tX - mu) / (sigma+VERY_SMALL_NUM)) < k, axis=1)
    return tX[keep_indices], y[keep_indices]