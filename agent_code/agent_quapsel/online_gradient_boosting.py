import numpy as np
import agent_code.agent1.train_params as tparam

class online_gradient_boost_regressor:
    """
    Gradient boost regressor class.
    """
    def __init__(self, weak_learner, learn_rate, **learner_params):
        """
        :param weak_learner: Weak learner used for gradient boost regression
        :param learn_rate: Learning rate
        :param *learner_params: All aditional parameters are passed to the weak learner.
        """
        # store parameters
        self.learner = weak_learner
        self.rate = learn_rate
        self.learner_params = learner_params

        # init estimator list
        self.estimators = []


    def predict(self, X):
        """
        Predict with current model.

        :param X: Feature matrix. Must meet requirements of weak learner.
        """
        N_samples = np.shape(X)[0]

        n_estimators = len(self.estimators)

        # default estimator return 0
        if n_estimators == 0:
            return np.zeros(N_samples)

        weight = 1
        predictions = np.empty((n_estimators, N_samples))

        # evaluate weak learners
        for i in range(n_estimators):
            predictions[i] = weight * self.estimators[i].predict(X)
            weight = self.rate * weight

        # predictions is sum
        predictions = np.sum(predictions, axis = 0)

        return predictions


    def fit_update(self, X, y):
        """
        Update Model by performing gradient boosting step.

        Params must meet requirements of weak learner.
        :param X: Feature matrix
        :param y: Responses
        """
        # compute residuals
        predictions = self.predict(X)
        residuals = y - predictions

        # learn correction estimator
        self.estimators.append(self.learner(**self.learner_params).fit(X, residuals))