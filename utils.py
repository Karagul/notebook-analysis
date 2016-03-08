import numpy as np
import matplotlib.pyplot as plt


def plot_deviance_and_importance(est, X_test, y_test, output_filename=None):
    """" Create plots of test/train deviance and feature importances. """

    test_score = np.zeros((est.n_estimators,), dtype=np.float64)

    for i, y_pred in enumerate(est.staged_predict(X_test)):
        test_score[i] = est.loss_(y_test, y_pred)

    # Plot Deviance
    plt.figure(figsize=(12, 6))
    plt.subplot(1, 2, 1)
    plt.title('Deviance')
    plt.plot(np.arange(est.n_estimators) + 1, est.train_score_, 'b-',
             label='Training Set Deviance')
    plt.plot(np.arange(est.n_estimators) + 1, test_score, 'r-',
             label='Test Set Deviance')
    plt.legend(loc='upper right')
    plt.xlabel('Boosting Iterations')
    plt.ylabel('Deviance')

    # Plot relative feature importance
    feature_importance = est.feature_importances_
    feature_importance = 100.0 * (feature_importance / feature_importance.max())
    sorted_idx = np.argsort(feature_importance)
    pos = np.arange(sorted_idx.shape[0]) + .5
    plt.subplot(1, 2, 2)
    plt.barh(pos, feature_importance[sorted_idx], align='center')
    plt.yticks(pos, X_test.columns[sorted_idx])
    plt.xlabel('Relative Importance')
    plt.title('Variable Importance')

    if output_filename:
        plt.savefig(output_filename)
