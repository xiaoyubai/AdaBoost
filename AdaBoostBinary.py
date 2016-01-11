import numpy as np
import pandas as pd
from sklearn.tree import DecisionTreeClassifier
from sklearn.base import clone
from sklearn.ensemble import AdaBoostClassifier


class AdaBoostBinaryClassifier(object):
    '''
    INPUT:
    - n_estimator (int)
      * The number of estimators to use in boosting
      * Default: 50
    - learning_rate (float)
      * Determines how fast the error would shrink
      * Lower learning rate means more accurate decision boundary,
        but slower to converge
      * Default: 1
    '''

    def __init__(self,
                 n_estimators=50,
                 learning_rate=1):

        self.base_estimator = DecisionTreeClassifier(max_depth=1)
        self.n_estimator = n_estimators
        self.learning_rate = learning_rate

        # Will be filled-in in the fit() step
        self.estimators_ = []
        self.estimator_weight_ = np.zeros(self.n_estimator, dtype=np.float)

    def fit(self, x, y):
        '''
        INPUT:
        - x: 2d numpy array, feature matrix
        - y: numpy array, labels
        Build the estimators for the AdaBoost estimator.
        '''
        sample_weight = np.ones(x.shape[0])/x.shape[0]
        for tree in range(self.n_estimator):
            estimator, sample_weight, estimator_weight= \
                self._boost(x,y, sample_weight)
            self.estimators_.append(estimator)
            self.estimator_weight_[tree]=estimator_weight



    def _boost(self, x, y, sample_weight):
        '''
        INPUT:
        - x: 2d numpy array, feature matrix
        - y: numpy array, labels
        - sample_weight: numpy array
        OUTPUT:
        - estimator: DecisionTreeClassifier
        - sample_weight: numpy array (updated weights)
        - estimator_weight: float (weight of estimator)
        Go through one iteration of the AdaBoost algorithm. Build one estimator.
        '''
        estimator = clone(self.base_estimator)
        dtc = estimator
        dtc.fit(x, y, sample_weight=sample_weight)
        pred_y = dtc.predict(x)
        indicator = np.ones(x.shape[0])*[pred_y!=y][0]
        err = np.dot(sample_weight, indicator) / np.sum(sample_weight)
        alpha = np.log((1-err)/err)
        new_sample_weight = sample_weight* np.exp(alpha*indicator)
        return estimator, new_sample_weight, alpha



    def predict(self, x):
        '''
        INPUT:
        - x: 2d numpy array, feature matrix
        OUTPUT:
        - labels: numpy array of predictions (0 or 1)
        '''
        predicts = []
        for estimator in self.estimators_:
            pred = estimator.predict(x)
            pred[pred==0] = -1
            predicts.append(pred)

        predicts = np.array(predicts)

        pr = np.sign(np.dot(self.estimator_weight_, predicts))
        pr[pr==-1] = 0
        return pr


    def score(self, x, y):
        '''
        INPUT:
        - x: 2d numpy array, feature matrix
        - y: numpy array, labels
        OUTPUT:
        - score: float (accuracy score between 0 and 1)
        '''
        #accuracy = TP+TN / n
        predictions = self.predict(x)
        n= x.shape[0]
        tp = np.sum(predictions * y)
        tn = np.sum((1-predictions)* (1-y))
        acc = (tp+tn)/n
        return acc

    def sklearn_AdaBoostClassifier(self, x_train, y_train, x_test, y_test):
        model = AdaBoostClassifier(self.base_estimator, self.n_estimator)
        model.fit(x_train, y_train)
        return model.score(x_test, y_test)
