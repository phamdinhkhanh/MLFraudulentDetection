# from pystacknet.pystacknet import StackNetClassifier
from sklearn.ensemble import RandomForestClassifier, ExtraTreesClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.neural_network import MLPClassifier
from sklearn import svm
from sklearn.impute import SimpleImputer
from sklearn.metrics import classification_report, accuracy_score, precision_recall_fscore_support
from sklearn.metrics import confusion_matrix
from sklearn.externals import joblib
from sklearn.model_selection import train_test_split
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import itertools
import os
import pandas as pd
import utils
import hyperameter as hp
from sklearn.model_selection import KFold
from datetime import datetime, timedelta

import logging
logging.basicConfig(filename='persistentvolume/log_{}.log'.format(datetime.strftime(datetime.now(), '%Y%m%d')),
                    filemode='a',format='%(asctime)s: %(levelname)s : %(message)s',
                    level = logging.INFO)
class Model(object):
    '''
    X: input of model
    y: target of model
    model: model created to training
    test_size: test size rate used to define train/test split
    '''
    def __init__(self, X_train = None, X_test = None, y_train = None, y_test = None, model = None, model_type = None, version = '0.0.2'):
        self.X_train = X_train
        self.X_test = X_test
        self.y_train = y_train
        self.y_test = y_test
        self.version = version
        self.model_type = model_type
        self._replace_nan()
        self.model = model

    def _replace_nan(self):
        if self.model_type in ['rd']:
            imp_zero = SimpleImputer(missing_values=np.nan, strategy='constant', fill_value=0)
            imp_zero.fit(self.X_train)
            self.X_train = imp_zero.transform(self.X_train)
            self.X_test = imp_zero.transform(self.X_test)

    def _train_cross_val(self, model_type = 'lgb', n_folds = 10, **kwargs):
        kfold = KFold(n_splits=n_folds, shuffle=True, random_state=12)
        valid_scores = []
        train_scores = []
        count = 0

        for train_idx, valid_idx in kfold.split(self.X_train):
            count += 1
            # Split train, valid
            train_features, train_labels = self.X_train[train_idx], self.y_train[train_idx]
            valid_features, valid_labels = self.X_train[valid_idx], self.y_train[valid_idx]

            valid_score = None
            train_score = None

            if self.model_type == 'lgb':
                self.model.fit(train_features, train_labels, eval_metric='auc',
                                   eval_set=[(valid_features, valid_labels), (train_features, train_labels)],
                                   eval_names=['valid', 'train'],
                                   early_stopping_rounds=100, verbose=200)
                valid_score = self.model.best_score_['valid']['auc']
                train_score = self.model.best_score_['train']['auc']
            elif self.model_type in ['rd', 'svm', 'dec']:
                self.model.fit(train_features, train_labels)
                y_pred_val = self.model.predict(valid_features)
                y_pred_train = self.model.predict(train_features)
                valid_score = accuracy_score(valid_labels, y_pred_val)
                train_score = accuracy_score(train_labels, y_pred_train)
            elif self.model_type == 'mlp':
                from keras.callbacks import EarlyStopping
                self.model.fit(train_features, train_labels,
                                   validation_data=[valid_features, valid_labels],
                                   callbacks=[EarlyStopping(monitor='val_loss', min_delta=0.001)])
                y_pred_train = [1 if item > 0.5 else 0 for item in self.model.predict(train_features)[:, 1]]
                y_pred_val = [1 if item > 0.5 else 0 for item in self.model.predict(valid_features)[:, 1]]
                valid_score = accuracy_score(valid_labels, y_pred_val)
                train_score = accuracy_score(train_labels, y_pred_train)
            else:
                logging.info('Have not yet support this model class!')

            valid_scores.append(valid_score)
            train_scores.append(train_score)
            logging.info('fold time: {}; train score: {}; valid score: {}'.format(count, valid_score, train_score))

        if self.model_type == 'rd':
            feature_importances = self.model.feature_importances_
            feature_importances = pd.DataFrame({'importance values': feature_importances})
            #print('kwargs', kwargs)
            final_features_fn = utils._add_month_filesave(hp.FINAL_FEATURES, kwargs['month'])
            final_features = utils.IOObject(final_features_fn)._load_pickle()
            final_features.remove('ORDER_CODE')
            feature_importances.index = final_features
            feature_importances = feature_importances.sort_values('importance values', ascending=False)
            importance_feature_fn = utils._add_month_filesave(hp.IMPORTANCE_FEATURE, kwargs['month'])
            utils.IOObject(importance_feature_fn)._save_pickle(list(feature_importances.index))
            logging.info('Already save important features into: {}'.format(importance_feature_fn))
            df_importance_feature_fn = utils._add_month_filesave(hp.DF_IMPORTANCE_FEATURE, kwargs['month'])
            utils.IOObject(df_importance_feature_fn)._save_pickle(feature_importances)
            logging.info('Already save dataframe important features into: {}'.format(df_importance_feature_fn))
        return train_scores, valid_scores

    def _train(self, X, y):
        self.model.fit(X, y)

    def _plot_prec_rec_curve(self, prec, rec, thres):
        '''
        :param prec: precision values
        :param rec: recall values
        :param thres: threshold corresponding with prec and rec
        :return: prec, rec curve
        '''
        plt.figure(figsize=(10, 8))
        plt.plot(thres, prec[:-1], 'b--', label='Precision')
        plt.plot(thres, rec[:-1], 'g-', label='Recall')
        plt.xlabel('Threshold')
        plt.ylabel('Probability')
        plt.title('Precsion vs Recall Curve')
        plt.legend()
        plt.show()

    def _plot_roc_curve(self, fpr, tpr, thres):
        '''
        :param fpr: false positive rate
        :param tpr: true positive rate
        :param thres: threshold corresponding to fpr and tpr
        :return: ROC curve
        '''
        plt.figure(figsize=(10, 8))
        plt.plot(fpr, tpr, 'b-', label='ROC')
        plt.plot([0, 1], [0, 1], '--')
        plt.axis([0, 1, 0, 1])
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title('ROC Curve')
        plt.show()

    def _plot_distribution_score(self, preds):
        sns.distplot(preds[:, 1], hist=True, kde=True,
                     bins=int(20), color='blue',
                     hist_kws={'edgecolor': 'black'})
        # Add labels
        plt.title('Distribution of Prob(y = 1)')
        plt.xlabel('Probability')
        plt.ylabel('Density')
        plt.show()

    def _predict_prob(self, X_input):
        '''
        :param X_input: X input
        :return: predicted class and prob(y = 1)
        '''
        proba = self.model.predict_proba(X_input)
        pred_class = np.argmax(proba, axis=1)
        return pred_class

    def _return_acc_report(self, clf_rep):
        out_dict = {
            "precision": list(clf_rep[0]),
            "recall": list(clf_rep[1]),
            "f1-score": list(clf_rep[2]),
            "support": list(clf_rep[3]),
            "index": [0, 1]
        }
        return out_dict

    def _return_acc_report(self, clf_rep):
        out_dict = {
            "precision": list(clf_rep[0]),
            "recall": list(clf_rep[1]),
            "f1-score": list(clf_rep[2]),
            "support": list(clf_rep[3]),
            "index": [0, 1]
        }
        return out_dict

    def _accuracy_score(self, data_set_type='test'):
        target_names = {'0': 'Thông thường', '1': 'Đầu cơ'}
        if data_set_type == 'test':
            y_pred = self._predict_prob(self.X_test)
            clf_report = precision_recall_fscore_support(self.y_test, y_pred)
            acc_report = self._return_acc_report(clf_report)
            acc = accuracy_score(self.y_test, y_pred)
            return {'accuracy_report': acc_report,
                    'accuracy': acc,
                    'target_names': target_names}
        elif data_set_type == 'train':
            y_pred = self._predict_prob(self.X_train)
            clf_report = precision_recall_fscore_support(self.y_train, y_pred)
            acc_report = self._return_acc_report(clf_report)
            acc = accuracy_score(self.y_train, y_pred)
            return {'accuracy_report': acc_report,
                    'accuracy': acc,
                    'target_names': target_names}

    def plot_confusion_matrix(self, y_test, y_pred,
                              normalize=False,
                              title='Confusion matrix',
                              cmap=plt.cm.Blues):
        """
        This function prints and plots the confusion matrix.
        Normalization can be applied by setting `normalize=True`.
        """
        target_names = ['Thông thường', 'Đầu cơ']
        cm = confusion_matrix(y_test, y_pred)
        plt.imshow(cm, interpolation='nearest', cmap=cmap)
        plt.title(title)
        plt.colorbar()
        tick_marks = np.arange(len(target_names))
        plt.xticks(tick_marks, target_names, rotation=0)
        plt.yticks(tick_marks, target_names)

        if normalize:
            cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
            # print("Normalized confusion matrix")
        else:
            1  # print('Confusion matrix, without normalization')

        # print(cm)

        thresh = cm.max() / 2.
        for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
            plt.text(j, i, cm[i, j],
                     horizontalalignment="center",
                     color="white" if cm[i, j] > thresh else "black")

        plt.tight_layout()
        plt.ylabel('True label')
        plt.xlabel('Predicted label')
        plt.show()

    def _predict(self, X_input):
        '''
        :param X_input: X input
        :return: predicted class and prob(y = 1)
        '''
        proba = self.model.predict_proba(X_input)
        pred_class = np.argmax(proba, axis=1)
        return pred_class, proba[:, 1]