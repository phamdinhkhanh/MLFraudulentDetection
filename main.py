from flask import Flask, request, render_template
from flask_restful import Resource, Api
from flask_api import FlaskAPI
# from waitress import serve
import logging
import sys
import json
import pandas as pd
from sklearn import svm
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from model import Model
from datetime import datetime
import utils
import os
import re
import lightgbm as lgb
import preprocessing
import preprocessing2
import pypyodbc
import time
from collections import defaultdict
import hyperameter as hp
import numpy as np

app = Flask(__name__)
app.logger.addHandler(logging.StreamHandler(sys.stdout))
app.logger.setLevel(logging.ERROR)
api = Api(app)

@app.route("/", methods = ['GET'])
def home():
    return 'Hello World!'

@app.route("/api/model/train", methods=['POST'])
def _train_model():
    # form = request.json
    form = request.get_json(force=True)
    logging.info('form response: {}'.format(form))
    model_type = form['model_type']
    month = form['month']
    logging.info('model type: {}'.format(model_type))
    model_init = None
    fn_model = None
    if model_type == 'lgb':
        fn_model = utils._add_month_filesave(hp.PREFIX_MODEL_FILE_LGB, month)
        # logging.info('Folder name: {}'.format(fn_model))
        if os.path.exists(fn_model):
            return 'Model lgb already trained!'
        else:
            # trái lại training
            model_init = lgb.LGBMClassifier(n_estimator=10000,
                                                objective='binary',
                                                class_weight='balanced',
                                                learning_rate=0.05,
                                                reg_alpha=0.1,
                                                reg_lambda=0.1,
                                                subsample=0.8,
                                                n_job=-1,
                                                random_state=12
                                                )
    elif model_type == 'rd':
        fn_model = utils._add_month_filesave(hp.PREFIX_MODEL_FILE_RDF, month)
        if os.path.exists(fn_model):
            return 'Model rd already trained!'
        else:
            model_init = RandomForestClassifier(n_estimators = 100, 
                                       max_depth = 5, 
                                       random_state = 123, 
                                       verbose = 1, 
                                       n_jobs = -1 
                                      )
    elif model_type == 'mlp':
        from keras.optimizers import Adam
        from keras.models import Sequential
        from keras.layers import Dense, Activation, BatchNormalization, Dropout
        fn_model = utils._add_month_filesave(hp.PREFIX_MODEL_FILE_MLP, month)
        if os.path.exists(fn_model):
            return 'Model mlp already trained!'
        else:
            model_init = Sequential([
                Dense(128, input_shape=(96,), activation='relu'),
                BatchNormalization(axis=-1),
                Dropout(0.8),
                Dense(64, activation='relu'),
                Dense(32, activation='relu'),
                Dropout(0.5),
                Dense(2),
                Activation('sigmoid'),
            ])

        model_init.compile(loss='sparse_categorical_crossentropy',
                               optimizer=Adam(),
                               metrics=['accuracy'])
    elif model_type == 'svm':
        fn_model = utils._add_month_filesave(hp.PREFIX_MODEL_FILE_SVM, month)
        if os.path.exists(fn_model):
            return 'Model svm already trained!'
        else:
            model_init = svm.SVC(gamma='scale', probability=True)
    elif model_type == 'dec':
        fn_model = utils._add_month_filesave(hp.PREFIX_MODEL_FILE_DEC, month)
        if os.path.exists(fn_model):
            return 'Model dec already trained!'
        else:
            model_init = DecisionTreeClassifier()
    X_train_fn = utils._add_month_filesave(hp.PREFIX_X_TRAIN, month)
    X_test_fn = utils._add_month_filesave(hp.PREFIX_X_TEST, month)
    y_train_fn = utils._add_month_filesave(hp.PREFIX_Y_TRAIN, month)
    y_test_fn = utils._add_month_filesave(hp.PREFIX_Y_TEST, month)
    scaler_fn = utils._add_month_filesave(hp.SCALER_FILE, month)
    final_features_fn = utils._add_month_filesave(hp.FINAL_FEATURES, month)
    df_summary_dummy_fn = utils._add_month_filesave(hp.DF_SUMMARY_DUMMY, month)
    corr_fn = utils._add_month_filesave(hp.DF_CORRELATION, month)
    if os.path.exists(X_train_fn) and os.path.exists(X_test_fn) and os.path.exists(y_train_fn) and os.path.exists(y_test_fn) and os.path.exists(final_features_fn) and os.path.exists(scaler_fn):
        X_train = utils.IOObject(X_train_fn)._load_pickle()
        X_test = utils.IOObject(X_test_fn)._load_pickle()
        y_train = utils.IOObject(y_train_fn)._load_pickle()
        y_test = utils.IOObject(y_test_fn)._load_pickle()
    else:
        dfTrainSum201906, X_train, X_test, y_train, y_test, scaler = preprocessing._read_data(is_train=True)
        utils.IOObject(final_features_fn)._save_pickle(list(dfTrainSum201906.columns))
        utils.IOObject(df_summary_dummy_fn)._save_pickle(dfTrainSum201906)
        utils.IOObject(y_test_fn)._save_pickle(y_test)
        utils.IOObject(X_train_fn)._save_pickle(X_train)
        utils.IOObject(X_test_fn)._save_pickle(X_test)
        utils.IOObject(y_train_fn)._save_pickle(y_train)
        utils.IOObject(y_test_fn)._save_pickle(y_test)
        utils.IOObject(scaler_fn)._save_pickle(scaler)
        # Lưu correlation giữa X_train với y_train
        # logging.info('dfTrainSum201906: {}'.format(dfTrainSum201906.columns))
        index_corr = list(dfTrainSum201906.columns)
        index_corr.remove('ORDER_CODE')
        corr = utils._correlation(X_train, y_train, final_features = index_corr)
        utils.IOObject(corr_fn)._save_pickle(corr)
    model = Model(X_train, X_test, y_train, y_test, model_init, model_type=model_type)
    model._train_cross_val(n_folds=5, month=month)
    utils.IOObject(fn_model)._save_pickle(model.model)
    back_test_result = model._accuracy_score()
    # logging.info('accuracy response: {}'.format(back_test_result))
    return json.dumps(back_test_result, ensure_ascii=False, cls = utils.NumpyEncoder)

@app.route("/api/model/trainv003", methods=['POST'])
def _train_model_window_1D():
    # form = request.json
    form = request.get_json(force=True)
    logging.info('form response: {}'.format(form))
    model_type = form['model_type']
    month = form['month']
    version = form['version']
    logging.info('model type: {}'.format(model_type))
    model_init = None
    fn_model = None
    if model_type == 'lgb':
        fn_model = utils._add_month_version_filesave(hp.PREFIX_MODEL_FILE_LGB, version, month)
        # logging.info('Folder name: {}'.format(fn_model))
        if os.path.exists(fn_model):
            return 'Model lgb already trained!'
        else:
            # trái lại training
            model_init = lgb.LGBMClassifier(n_estimator=10000,
                                                objective='binary',
                                                class_weight='balanced',
                                                learning_rate=0.05,
                                                reg_alpha=0.1,
                                                reg_lambda=0.1,
                                                subsample=0.8,
                                                n_job=-1,
                                                random_state=12
                                                )
    elif model_type == 'rd':
        fn_model = utils._add_month_version_filesave(hp.PREFIX_MODEL_FILE_RDF, version, month)
        if os.path.exists(fn_model):
            return 'Model rd already trained!'
        else:
            model_init = RandomForestClassifier(n_estimators = 100,
                                       max_depth = 5,
                                       random_state = 123,
                                       verbose = 1,
                                       n_jobs = -1
                                      )
    elif model_type == 'mlp':
        from keras.optimizers import Adam
        from keras.models import Sequential
        from keras.layers import Dense, Activation, BatchNormalization, Dropout
        fn_model = utils._add_month_version_filesave(hp.PREFIX_MODEL_FILE_MLP, version, month)
        if os.path.exists(fn_model):
            return 'Model mlp already trained!'
        else:
            model_init = Sequential([
                Dense(128, input_shape=(96,), activation='relu'),
                BatchNormalization(axis=-1),
                Dropout(0.8),
                Dense(64, activation='relu'),
                Dense(32, activation='relu'),
                Dropout(0.5),
                Dense(2),
                Activation('sigmoid'),
            ])

        model_init.compile(loss='sparse_categorical_crossentropy',
                               optimizer=Adam(),
                               metrics=['accuracy'])
    elif model_type == 'svm':
        fn_model = utils._add_month_version_filesave(hp.PREFIX_MODEL_FILE_SVM, version, month)
        if os.path.exists(fn_model):
            return 'Model svm already trained!'
        else:
            model_init = svm.SVC(gamma='scale', probability=True)
    elif model_type == 'dec':
        fn_model = utils._add_month_version_filesave(hp.PREFIX_MODEL_FILE_DEC, version, month)
        if os.path.exists(fn_model):
            return 'Model dec already trained!'
        else:
            model_init = DecisionTreeClassifier()
    X_train_fn = utils._add_month_filesave(hp.PREFIX_X_TRAIN, month)
    X_test_fn = utils._add_month_filesave(hp.PREFIX_X_TEST, month)
    y_train_fn = utils._add_month_filesave(hp.PREFIX_Y_TRAIN, month)
    y_test_fn = utils._add_month_filesave(hp.PREFIX_Y_TEST, month)
    scaler_fn = utils._add_month_filesave(hp.SCALER_FILE, month)
    final_features_fn = utils._add_month_filesave(hp.FINAL_FEATURES, month)
    df_summary_dummy_fn = utils._add_month_filesave(hp.DF_SUMMARY_DUMMY, month)
    corr_fn = utils._add_month_filesave(hp.DF_CORRELATION, month)
    if os.path.exists(X_train_fn) and os.path.exists(X_test_fn) and os.path.exists(y_train_fn) and os.path.exists(y_test_fn) and os.path.exists(final_features_fn) and os.path.exists(scaler_fn):
        X_train = utils.IOObject(X_train_fn)._load_pickle()
        X_test = utils.IOObject(X_test_fn)._load_pickle()
        y_train = utils.IOObject(y_train_fn)._load_pickle()
        y_test = utils.IOObject(y_test_fn)._load_pickle()
    else:
        df1M = preprocessing2.preprocess_order_1M(hp.previous_30_days, hp.today)
        dfTrainSum201906, X_train, X_test, y_train, y_test, scaler = preprocessing2._train_test_split(df1M)
        utils.IOObject(final_features_fn)._save_pickle(list(dfTrainSum201906.columns))
        utils.IOObject(df_summary_dummy_fn)._save_pickle(dfTrainSum201906)
        utils.IOObject(y_test_fn)._save_pickle(y_test)
        utils.IOObject(X_train_fn)._save_pickle(X_train)
        utils.IOObject(X_test_fn)._save_pickle(X_test)
        utils.IOObject(y_train_fn)._save_pickle(y_train)
        utils.IOObject(y_test_fn)._save_pickle(y_test)
        utils.IOObject(scaler_fn)._save_pickle(scaler)
        # Lưu correlation giữa X_train với y_train
        # logging.info('dfTrainSum201906: {}'.format(dfTrainSum201906.columns))
        index_corr = list(dfTrainSum201906.columns)
        index_corr.remove('ORDER_CODE')
        corr = utils._correlation(X_train, y_train, final_features = index_corr)
        utils.IOObject(corr_fn)._save_pickle(corr)
    model = Model(X_train, X_test, y_train, y_test, model_init, model_type=model_type, version='0.0.3')
    model._train_cross_val(n_folds=5, month=month)
    utils.IOObject(fn_model)._save_pickle(model.model)
    back_test_result = model._accuracy_score()
    # logging.info('accuracy response: {}'.format(back_test_result))
    return json.dumps(back_test_result, ensure_ascii=False, cls = utils.NumpyEncoder)

@app.route("/api/predict", methods = ['POST'])
def predictSpeculation():
    time.sleep(1)
    if request.method == 'POST':
        # form = request.json
        form = request.get_json(force=True)
        # data_obs = utils._input_data_dict(form)
        order = form['order_code']
        month = form['month']
        start_time = datetime.now()
        ORDER_CODE, dfTrainSum201906, X = preprocessing._read_data(is_train=False, order_id=order)
        end_time = datetime.now()
        logging.info('1. time of loading data: {}'.format(end_time-start_time))
        fn_model = utils._add_month_filesave(hp.PREFIX_MODEL_FILE_LGB, month)
        start_time = datetime.now()
        model = utils.IOObject(fn_model)._load_pickle()
        end_time = datetime.now()
        logging.info('2. time of loading model: {}'.format(end_time-start_time))
        # logging.info('shape X: {}'.format(X.shape))
        start_time = datetime.now()
        y_prob, y_class = utils._predict_prob_class(model, thres=0.5, X=X)
        # delete X
        utils._gc_collect(X)
        # logging.info('numpy y_prob dtypes: {}'.format(y_prob.dtype))
        dfResult = pd.DataFrame({'ORDER_CODE': ORDER_CODE, 'y_prob': y_prob, 'y_class': y_class})
        # details_sort_importance
        result = utils._parse_predict_one_order(dfResult, dfTrainSum201906, order, month)
        # delete dfTrainSum201906
        utils._gc_collect(dfTrainSum201906)
        end_time = datetime.now()
        logging.info('3. time of prediction: {}'.format(end_time - start_time))
        return json.dumps(result, ensure_ascii=False, cls = utils.NumpyEncoder)

@app.route("/api/predict/importance_percentage", methods = ['POST'])
def predictSpeculationImportantPercentage():
    time.sleep(1)
    if request.method == 'POST':
        # form = request.json
        form = request.get_json(force=True)
        # data_obs = utils._input_data_dict(form)
        order = form['order_code']
        month = form['month']
        start_time = datetime.now()
        ORDER_CODE, dfTrainSum201906, X = preprocessing._read_data(is_train=False, order_id=order)
        end_time = datetime.now()
        logging.info('1. time of loading data: {}'.format(end_time-start_time))
        fn_model = utils._add_month_filesave(hp.PREFIX_MODEL_FILE_LGB, month)
        start_time = datetime.now()
        model = utils.IOObject(fn_model)._load_pickle()
        end_time = datetime.now()
        logging.info('2. time of loading model: {}'.format(end_time-start_time))
        # logging.info('shape X: {}'.format(X.shape))
        start_time = datetime.now()
        y_prob, y_class = utils._predict_prob_class(model, thres=0.5, X=X)
        # delete X
        # utils._gc_collect(X)
        # logging.info('numpy y_prob dtypes: {}'.format(y_prob.dtype))
        dfResult = pd.DataFrame({'ORDER_CODE': ORDER_CODE, 'y_prob': y_prob, 'y_class': y_class})
        # Lấy index order id
        idx = list(np.where(np.array(ORDER_CODE) == order)[0])[0]
        # load df_corr and df_importance_features
        df_corr_fn = utils._add_month_filesave(hp.DF_CORRELATION, month)
        df_corr = utils.IOObject(df_corr_fn)._load_pickle()
        df_features_important_fn = utils._add_month_filesave(hp.DF_IMPORTANCE_FEATURE, month)
        df_features_important = utils.IOObject(df_features_important_fn)._load_pickle()
        # utils._parse_predict_percentage_important(, df_corr=df_corr, df_features_important=df_features_important)
        logging.info('X.shape: {}'.format(X.shape))
        logging.info('df_corr.shape: {}'.format(df_corr.shape))
        # logging.info('df_features_important.shape: {}'.format(df_features_important.shape))
        X_input = X[idx, :].T
        result = utils._parse_predict_percentage_important_variables(
            dfResult = dfResult, dataset = dfTrainSum201906,
            order = order, month = month,
            df_corr = df_corr, df_features_important = df_features_important,
            X_input = X_input,
            modelVersion = "0.0.2"
        )
        # result = utils._parse_predict_percentage_important(dfResult, dfTrainSum201906, order, month)
        # delete dfTrainSum201906
        utils._gc_collect(dfTrainSum201906)
        end_time = datetime.now()
        logging.info('3. time of prediction: {}'.format(end_time - start_time))
        return json.dumps(result, ensure_ascii=False, cls = utils.NumpyEncoder)

@app.route("/api/predict/importance_percentage_v003", methods = ['POST'])
def predictSpeculationImportantPercentageV003():
    time.sleep(1)
    if request.method == 'POST':
        # form = request.json
        form = request.get_json(force=True)
        # data_obs = utils._input_data_dict(form)
        order = form['order_code']
        month = form['month']
        version = form['version']
        start_time = datetime.now()
        ORDER_CODE, dfTrainSum201906, X = preprocessing2.preprocess_orderids(order_id=order)
        end_time = datetime.now()
        logging.info('1. time of loading data: {}'.format(end_time-start_time))
        fn_model = utils._add_month_version_filesave(hp.PREFIX_MODEL_FILE_LGB, version, month)
        start_time = datetime.now()
        model = utils.IOObject(fn_model)._load_pickle()
        end_time = datetime.now()
        logging.info('2. time of loading model: {}'.format(end_time-start_time))
        # logging.info('shape X: {}'.format(X.shape))
        start_time = datetime.now()
        y_prob, y_class = utils._predict_prob_class(model, thres=0.5, X=X)
        # delete X
        # utils._gc_collect(X)
        # logging.info('numpy y_prob dtypes: {}'.format(y_prob.dtype))
        dfResult = pd.DataFrame({'ORDER_CODE': ORDER_CODE, 'y_prob': y_prob, 'y_class': y_class})
        # Lấy index order id
        idx = list(np.where(np.array(ORDER_CODE) == order)[0])[0]
        # load df_corr and df_importance_features
        df_corr_fn = utils._add_month_filesave(hp.DF_CORRELATION, month)
        df_corr = utils.IOObject(df_corr_fn)._load_pickle()
        df_features_important_fn = utils._add_month_filesave(hp.DF_IMPORTANCE_FEATURE, month)
        df_features_important = utils.IOObject(df_features_important_fn)._load_pickle()
        # utils._parse_predict_percentage_important(, df_corr=df_corr, df_features_important=df_features_important)
        logging.info('X.shape: {}'.format(X.shape))
        logging.info('df_corr.shape: {}'.format(df_corr.shape))
        # logging.info('df_features_important.shape: {}'.format(df_features_important.shape))
        X_input = X[idx, :].T
        result = utils._parse_predict_percentage_important_variables(
            dfResult = dfResult, dataset = dfTrainSum201906,
            order = order, month = month,
            df_corr = df_corr, df_features_important = df_features_important,
            X_input = X_input,
            modelVersion = "0.0.3"
        )
        # result = utils._parse_predict_percentage_important(dfResult, dfTrainSum201906, order, month)
        # delete dfTrainSum201906
        utils._gc_collect(dfTrainSum201906)
        end_time = datetime.now()
        logging.info('3. time of prediction: {}'.format(end_time - start_time))
        return json.dumps(result, ensure_ascii=False, cls = utils.NumpyEncoder)

@app.route("/api/model/check_complete", methods = ['POST'])
def checkCompletedTrain():
    if request.method == 'POST':
        # form = request.json
        form = request.get_json(force=True)
        # data_obs = utils._input_data_dict(form)
        month = form['month']
        models = form['models']
        version = form['version']
        checkFiles = []
        for model in models:
            filename = model + '_classifier_all'
            filename = utils._add_month_version_filesave(filename, version, month)
            filename = os.path.join('persistentvolume/models/', filename)
            checkFile = utils.IOObject(filename)._check_file()
            checkFiles += [checkFile]
        if all(checkFiles):
            result = {
                'month': month,
                'version': version,
                'model_completed': True
            }
        else:
            result = {
                'month': month,
                'version': version,
                'model_completed': False
            }
        return json.dumps(result, ensure_ascii=False, cls=utils.NumpyEncoder)

if __name__ == "__main__":
    app.run(debug=True, host=hp.IP)
    # serve(app, host=hp.IP)
    # _train_model()
