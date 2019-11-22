from sklearn.externals import joblib
import pandas as pd
import numpy as np
import gc
from datetime import datetime
import hyperameter as hp
import os
import json
import logging
import pickle
from collections import defaultdict
import cx_Oracle

logging.basicConfig(filename='persistentvolume/log_{}.log'.format(datetime.strftime(datetime.now(), '%Y%m%d')),
                    filemode='a',format='%(asctime)s: %(levelname)s : %(message)s',
                    level = logging.INFO)

def _merge_summary_dimension(dfSummaryTable, dfTrainSum201906, columns, join_column):
    '''
    :param dfSummaryTable: pivot_table after group by column
    :param dfTrainSum201906: train summary table need to merge dfSummaryTable into
    :param columns: list columns in dfSummaryTable
    :param join_column: column used to join two table
    :param value: value of new groupby field in corresponding with new order. Only used for new order ID.
    :return:
    '''
    dfSummaryTable.columns = columns
    dfTrainSum201906 = pd.merge(dfTrainSum201906, dfSummaryTable, left_on=join_column,
                                right_on=join_column, how='left')
    return dfTrainSum201906


def _sql_query_prev_30Days(table, is_train = False):
    if is_train:
        query = "SELECT ORDER_DATE, ORDER_CODE,  PAYMENTMODES, PAYMENTSTATUS, EMAIL_ACCOUNT, USER_ID, PHONE_DELIVERY, EMAIL_DELIVERY, \
                P_STREETNUMBER, STREET, WARD, VNDISTRICT, PROVINCE, OFFER_CODE, OFFER_NAME, MERCHANTCODE, MERCHANTNAME, WAREHOUSE, P_QUANTITY, \
                AFF_COMMISSION, AFF_PARTNERNAME, AFF_UTMSOURCE, SUB2_MAIN, SUB3_MAIN, SUB4_MAIN, SUB5_MAIN, SUB6_MAIN, SUB7_MAIN, CODE2_MAIN, \
                CODE3_MAIN, CODE4_MAIN, CODE5_MAIN, CODE6_MAIN, CODE7_MAIN, Item_PK, OFFER_FINAL_PRICE, OFFER_PRICE, P_SUBTOTAL, P_TOTALPRICE, \
                DISCOUNT_AMOUNT, P_TOTALORDERAMOUNT, ORDER_USER_VINIDACCOUNT, ORDER_VINIDNUMBER, ORDER_CUSTOMER_TYPE, SALES_APPLICATION, IP_ADDRESS, \
                FRAUD_SaveValue_Check, TTKD FROM " + table + " WHERE ORDER_CUSTOMER_TYPE = 'B2C' AND ORDER_STATUS != 'NEW'"
    else:
        query = "SELECT ORDER_DATE, ORDER_CODE,  PAYMENTMODES, PAYMENTSTATUS, EMAIL_ACCOUNT, USER_ID, PHONE_DELIVERY, EMAIL_DELIVERY, \
                P_STREETNUMBER, STREET, WARD, VNDISTRICT, PROVINCE, OFFER_CODE, OFFER_NAME, MERCHANTCODE, MERCHANTNAME, WAREHOUSE, P_QUANTITY, \
                AFF_COMMISSION, AFF_PARTNERNAME, AFF_UTMSOURCE, SUB2_MAIN, SUB3_MAIN, SUB4_MAIN, SUB5_MAIN, SUB6_MAIN, SUB7_MAIN, CODE2_MAIN, \
                CODE3_MAIN, CODE4_MAIN, CODE5_MAIN, CODE6_MAIN, CODE7_MAIN, Item_PK, OFFER_FINAL_PRICE, OFFER_PRICE, P_SUBTOTAL, P_TOTALPRICE, \
                DISCOUNT_AMOUNT, P_TOTALORDERAMOUNT, ORDER_USER_VINIDACCOUNT, ORDER_VINIDNUMBER, ORDER_CUSTOMER_TYPE, SALES_APPLICATION, IP_ADDRESS, \
                TTKD FROM " + table + " WHERE ORDER_CUSTOMER_TYPE = 'B2C' AND ORDER_STATUS != 'NEW'"
    return query

def _sql_query_OrderID(table, order_id):
    query = "SELECT ORDER_CODE, PHONE_DELIVERY, IP_ADDRESS, ORDER_VINIDNUMBER FROM " + table + " WHERE ORDER_CODE = '" + str(order_id) + "'"
    return query

def _getAllOrder1M(columns = hp.COLUMNS_QUERY_NEW_ORDER):
    with cx_Oracle.connect(hp.constring) as cnxn:
        cur = cnxn.cursor()
        o_MYCURSOR = cur.var(cx_Oracle.CURSOR)
        result = cur.callproc('HYBRIS.VIEW_GETALLINFO_ORDER_1M', [o_MYCURSOR])
        df = pd.DataFrame(result[0].fetchall(), columns=columns)
    return df

def _getAllOrder1MByPhoneIpAcc(p_phone = None, p_ip = None, p_acc = None, columns = hp.COLUMNS_QUERY_NEW_ORDER):
    with cx_Oracle.connect(hp.constring) as cnxn:
        cur = cnxn.cursor()
        o_MYCURSOR = cur.var(cx_Oracle.CURSOR)
        result = cur.callproc('HYBRIS.View_Order_1M_By_Phone_IP_Acc', [p_phone, p_ip, p_acc, o_MYCURSOR])
        df = pd.DataFrame(result[3].fetchall(), columns = columns)
    return df

def _getAllOrder1MTraining(p_date_from = None, p_date_to = None, columns = hp.COLUMNS_QUERY):
    with cx_Oracle.connect(hp.constring) as cnxn:
        cur = cnxn.cursor()
        o_MYCURSOR = cur.var(cx_Oracle.CURSOR)
        result = cur.callproc('HYBRIS.GETALLINFO_TRAINING_BY_DATE', [p_date_from, p_date_to, o_MYCURSOR])
        df = pd.DataFrame(result[2].fetchall(), columns = columns)
    return df

def _getInfoByOrder(p_orderid = None):
    with cx_Oracle.connect(hp.constring) as cnxn:
        cur = cnxn.cursor()
        o_MYCURSOR = cur.var(cx_Oracle.CURSOR)
        result = cur.callproc('HYBRIS.VIEW_GET_P_I_V_INFO_BYORDER', [p_orderid, o_MYCURSOR])
        columns = ['ORDER_CODE', 'PHONE_DELIVERY', 'IP_ADDRESS', 'ORDER_VINIDNUMBER']
        df = pd.DataFrame(result[1].fetchall(), columns = columns).head(1)
    return df

def _getAllOrderTodayByOrder(p_orderid = None, columns = hp.COLUMNS_QUERY_NEW_ORDER):
    with cx_Oracle.connect(hp.constring) as cnxn:
        cur = cnxn.cursor()
        o_MYCURSOR = cur.var(cx_Oracle.CURSOR)
        result = cur.callproc('HYBRIS.VIEW_GETALLINFO_BYORDER_TODAY', [p_orderid, o_MYCURSOR])
        df = pd.DataFrame(result[1].fetchall(), columns=columns)
        _change_type_col('P_QUANTITY', df, is_replace_na=False)
        _change_type_col('AFF_COMMISSION', df, is_replace_na=True)
        _change_type_col('OFFER_FINAL_PRICE', df, is_replace_na=False)
        _change_type_col('OFFER_PRICE', df, is_replace_na=False)
        _change_type_col('P_SUBTOTAL', df, is_replace_na=False)
        _change_type_col('P_TOTALPRICE', df, is_replace_na=False)
        _change_type_col('DISCOUNT_AMOUNT', df, is_replace_na=False)
        _change_type_col('P_TOTALORDERAMOUNT', df, is_replace_na=False)
    return df

def _sql_query_new_order(table, phone, ip, acc):
    #\\TODO check với điều kiện phone is None, ip is None và acc is None
    if (ip is not None) and (acc is not None):
        query = "SELECT ORDER_DATE, ORDER_CODE,  PAYMENTMODES, PAYMENTSTATUS, EMAIL_ACCOUNT, USER_ID, PHONE_DELIVERY, EMAIL_DELIVERY, P_STREETNUMBER, \
                STREET, WARD, VNDISTRICT, PROVINCE, OFFER_CODE, OFFER_NAME, MERCHANTCODE, MERCHANTNAME, WAREHOUSE, P_QUANTITY, AFF_COMMISSION, \
                AFF_PARTNERNAME, AFF_UTMSOURCE, SUB2_MAIN, SUB3_MAIN, SUB4_MAIN, SUB5_MAIN, SUB6_MAIN, SUB7_MAIN, CODE2_MAIN, CODE3_MAIN, CODE4_MAIN, \
                CODE5_MAIN, CODE6_MAIN, CODE7_MAIN, Item_PK, OFFER_FINAL_PRICE, OFFER_PRICE, P_SUBTOTAL, P_TOTALPRICE, DISCOUNT_AMOUNT, P_TOTALORDERAMOUNT, \
                ORDER_USER_VINIDACCOUNT, ORDER_VINIDNUMBER, ORDER_CUSTOMER_TYPE, SALES_APPLICATION, IP_ADDRESS, TTKD FROM " + table + " WHERE PHONE_DELIVERY = '" \
                + str(phone) + "' AND IP_ADDRESS = '" + str(ip) + "' AND ORDER_VINIDNUMBER = '" + str(acc) + "'"
    elif (ip is None) and (acc is not None):
        query = "SELECT ORDER_DATE, ORDER_CODE,  PAYMENTMODES, PAYMENTSTATUS, EMAIL_ACCOUNT, USER_ID, PHONE_DELIVERY, EMAIL_DELIVERY, P_STREETNUMBER, \
             STREET, WARD, VNDISTRICT, PROVINCE, OFFER_CODE, OFFER_NAME, MERCHANTCODE, MERCHANTNAME, WAREHOUSE, P_QUANTITY, AFF_COMMISSION, \
             AFF_PARTNERNAME, AFF_UTMSOURCE, SUB2_MAIN, SUB3_MAIN, SUB4_MAIN, SUB5_MAIN, SUB6_MAIN, SUB7_MAIN, CODE2_MAIN, CODE3_MAIN, CODE4_MAIN, \
             CODE5_MAIN, CODE6_MAIN, CODE7_MAIN, Item_PK, OFFER_FINAL_PRICE, OFFER_PRICE, P_SUBTOTAL, P_TOTALPRICE, DISCOUNT_AMOUNT, P_TOTALORDERAMOUNT, \
             ORDER_USER_VINIDACCOUNT, ORDER_VINIDNUMBER, ORDER_CUSTOMER_TYPE, SALES_APPLICATION, IP_ADDRESS, TTKD FROM " + table + " WHERE PHONE_DELIVERY = '" \
                + str(phone) + "' AND ORDER_VINIDNUMBER = '" + str(acc) + "'"
    elif (ip is None) and (acc is None):
        query = "SELECT ORDER_DATE, ORDER_CODE,  PAYMENTMODES, PAYMENTSTATUS, EMAIL_ACCOUNT, USER_ID, PHONE_DELIVERY, EMAIL_DELIVERY, P_STREETNUMBER, \
         STREET, WARD, VNDISTRICT, PROVINCE, OFFER_CODE, OFFER_NAME, MERCHANTCODE, MERCHANTNAME, WAREHOUSE, P_QUANTITY, AFF_COMMISSION, \
         AFF_PARTNERNAME, AFF_UTMSOURCE, SUB2_MAIN, SUB3_MAIN, SUB4_MAIN, SUB5_MAIN, SUB6_MAIN, SUB7_MAIN, CODE2_MAIN, CODE3_MAIN, CODE4_MAIN, \
         CODE5_MAIN, CODE6_MAIN, CODE7_MAIN, Item_PK, OFFER_FINAL_PRICE, OFFER_PRICE, P_SUBTOTAL, P_TOTALPRICE, DISCOUNT_AMOUNT, P_TOTALORDERAMOUNT, \
         ORDER_USER_VINIDACCOUNT, ORDER_VINIDNUMBER, ORDER_CUSTOMER_TYPE, SALES_APPLICATION, IP_ADDRESS, TTKD FROM " + table + " WHERE PHONE_DELIVERY = '" \
         + str(phone) + "'"
    elif (ip is not None) and (acc is None):
        query = "SELECT ORDER_DATE, ORDER_CODE,  PAYMENTMODES, PAYMENTSTATUS, EMAIL_ACCOUNT, USER_ID, PHONE_DELIVERY, EMAIL_DELIVERY, P_STREETNUMBER, \
                 STREET, WARD, VNDISTRICT, PROVINCE, OFFER_CODE, OFFER_NAME, MERCHANTCODE, MERCHANTNAME, WAREHOUSE, P_QUANTITY, AFF_COMMISSION, \
                 AFF_PARTNERNAME, AFF_UTMSOURCE, SUB2_MAIN, SUB3_MAIN, SUB4_MAIN, SUB5_MAIN, SUB6_MAIN, SUB7_MAIN, CODE2_MAIN, CODE3_MAIN, CODE4_MAIN, \
                 CODE5_MAIN, CODE6_MAIN, CODE7_MAIN, Item_PK, OFFER_FINAL_PRICE, OFFER_PRICE, P_SUBTOTAL, P_TOTALPRICE, DISCOUNT_AMOUNT, P_TOTALORDERAMOUNT, \
                 ORDER_USER_VINIDACCOUNT, ORDER_VINIDNUMBER, ORDER_CUSTOMER_TYPE, SALES_APPLICATION, IP_ADDRESS, TTKD FROM " + table + " WHERE PHONE_DELIVERY = '" \
                + str(phone) + "' AND IP_ADDRESS = '" + str(ip) + "'"
    return query

class IOObject(object):
    def __init__(self, filename):
        self.filename = filename

    def _save_pickle(self, obj):
        with open(self.filename, "wb") as fp:  # Pickling
            pickle.dump(obj, fp)

    def _open_pickle(self):
        with open(self.filename, "rb") as fp:  # Unpickling
            obj = pickle.load(fp)
        return obj

    def _save_json(self, obj):
        with open(self.filename, "wb") as fp:
            json.dump(obj, fp)

    def _open_json(self):
        with open(self.filename, "rb") as fp:
            obj = json.loads(fp.read())
        return obj

    def _load_pickle(self):
        if os.path.exists(self.filename):
            obj = joblib.load(self.filename)
        else:
            logging.info('Filename is not existed!')
            return None
        return obj

    def _dump_pickle(self, obj, replace=False):
        if (os.path.exists(self.filename)) & (not replace):
            logging.info('Filename is existed!')
        else:
            joblib.dump(obj, self.filename)

    def _check_file(self):
        if os.path.exists(self.filename):
            logging.info('Check filename is existed!')
            return True
        else:
            return False

def _parse_predict_one_order2(dfResult, dataset, order, month):
    dfResult = dfResult[dfResult['ORDER_CODE'] == order]
    dataset = dataset[dataset['ORDER_CODE'] == order]
    details_sort_importance = []
    fraudReason = defaultdict(list)
    df_final_feature_fn = _add_month_filesave(hp.DF_FINAL_FEATURE, month)
    df_features = pd.read_pickle(df_final_feature_fn)
    for i, row in df_features[['VARIABLE_GROUP', 'VARIABLE_NAMES', 'FEATURES_NAME', 'MEANING', 'VALUES', 'VALUES_TYPES']].iterrows():
        if (row['VALUES_TYPES'] == "string") & (dataset[row['FEATURES_NAME']].values == 1):
            detail = str(row['VARIABLE_NAMES']) + " - " + str(row['MEANING']) + " : " + str(row['VALUES'])
            fraudGroup = str(row['VARIABLE_GROUP'])
            fraudReason[fraudGroup] += [detail]
            details_sort_importance.append(detail)

        if row['VALUES_TYPES'] == 'float':
            detail = str(row['VARIABLE_NAMES']) + " - " + str(row['MEANING']) + " : " + str(dataset[row['FEATURES_NAME']].values[0])
            fraudGroup = str(row['VARIABLE_GROUP'])
            fraudReason[fraudGroup] += [detail]
            details_sort_importance.append(detail)

        if (row['VALUES_TYPES'] == 'boolean') & (dataset[row['FEATURES_NAME']].values == 1):
            detail = str(row['VARIABLE_NAMES']) + " - " + str(row['MEANING']) + " : " + str(dataset[row['FEATURES_NAME']].values[0])
            fraudGroup = str(row['VARIABLE_GROUP'])
            fraudReason[fraudGroup] += [detail]
            details_sort_importance.append(detail)

    # Chuyển về đúng định dạng theo request
    # {
    #     "SOID": "9044224931",
    #     "probability": 0.9505635179688862,
    #     "predictedClass": 1,
    #     "modelVersion": "0.0.1"
    #     "fraudReason": [
    #        "fraudGroup":"OrderAmount",
    #        "details": "Paid VINID 5000; TotalAmount 200000"
    # ],
    # // repeat if multiple fraudGroup
    # }

    fraudReasonList = []
    for item in fraudReason.items():
        fraudReasonList.append({
            "fraudGroup":item[0],
            "details":item[1]
        })

    result = {
        "SOID":order,
        "probability":dfResult['y_prob'].values,
        "predictedClass": dfResult['y_class'].values,
        "modelVersion":"0.0.2",
        "fraudReason":fraudReasonList,
        "fraudReasonSortImportance":details_sort_importance
    }
    return result

def _parse_predict_one_order(dfResult, dataset, order, month):
    dfResult = dfResult[dfResult['ORDER_CODE'] == order]
    dataset = dataset[dataset['ORDER_CODE'] == order]
    details_sort_importance = []
    fraudReason = defaultdict(list)
    df_final_feature_fn = _add_month_filesave(hp.DF_FINAL_FEATURE, month)
    df_features = pd.read_pickle(df_final_feature_fn)
    df_features['VARIABLE_NAMES'] = df_features['VARIABLE_NAMES'].str.strip()
    df_features['FEATURES_NAME'] = df_features['FEATURES_NAME'].str.strip()
    dict_features = hp.RETURN_HYBRIS
    dict_var_groups = hp.RETURN_MAP_VARGROUP
    for i, row in df_features[
        ['VARIABLE_GROUP', 'VARIABLE_NAMES', 'FEATURES_NAME', 'MEANING', 'VALUES', 'VALUES_TYPES']].iterrows():
        if row['VARIABLE_NAMES'] in dict_features:
            try:
                if (row['VALUES_TYPES'] == "string") & (dataset[row['FEATURES_NAME']].values == 1):
                    detail = str(dict_features[row['VARIABLE_NAMES']]) + " : " + str(row['VALUES'])
                    fraudGroup = str(dict_var_groups[row['VARIABLE_GROUP']])
                    fraudReason[fraudGroup] += [detail]
                    details_sort_importance.append(detail)

                if row['VALUES_TYPES'] == 'float':
                    detail = str(dict_features[row['VARIABLE_NAMES']]) + " : " + str(
                        dataset[row['FEATURES_NAME']].values[0])
                    fraudGroup = str(dict_var_groups[row['VARIABLE_GROUP']])
                    fraudReason[fraudGroup] += [detail]
                    details_sort_importance.append(detail)

                if (row['VALUES_TYPES'] == 'boolean') & (dataset[row['FEATURES_NAME']].values == 1):
                    detail = str(dict_features[row['VARIABLE_NAMES']]) + " : " + str(
                        dataset[row['FEATURES_NAME']].values[0])
                    fraudGroup = dict_var_groups[row['VARIABLE_GROUP']]
                    fraudReason[fraudGroup] += [detail]
                    details_sort_importance.append(detail)
            except:
                logging.info('Not exist: {}'.format(row['FEATURES_NAME']))
    # Chuyển về đúng định dạng theo request
    # {
    #     "SOID": "9044224931",
    #     "probability": 0.9505635179688862,
    #     "predictedClass": 1,
    #     "modelVersion": "0.0.1"
    #     "fraudReason": [
    #        "fraudGroup":"OrderAmount",
    #        "details": "Paid VINID 5000; TotalAmount 200000"
    # ],
    # // repeat if multiple fraudGroup
    # }

    fraudReasonList = []
    for item in fraudReason.items():
        fraudReasonList.append({
            "fraudGroup": item[0],
            "details": item[1]
        })

    result = {
        "SOID": order,
        "probability": dfResult['y_prob'].values,
        "predictedClass": dfResult['y_class'].values,
        "modelVersion": "0.0.2",
        "fraudReason": fraudReasonList
        # "fraudReasonSortImportance": details_sort_importance
    }
    return result


def _parse_predict_percentage_important_features(dfResult, dataset, order, month, df_corr, X_input, df_features_important = None):
    '''
    :param dfResult: Kết quả trả về từ API
    :param dataset: Dữ liệu gốc của đơn hàng
    :param order: Mã đơn hàng
    :param month: Tháng
    :param df_corr: Bảng correction giữa các biến
    :param df_features_important: Bảng features important lấy từ random forest model
    :param X_input: Đầu vào của mô hình
    :return: result of models
    '''
    # 1. Tính mức độ quan trọng important_percentage
    df_weight, order_columns_weight = _cal_weight_percent3(df_corr, X_input)
    # df_weight, order_columns_weight = _cal_weight_percent2(df_corr, df_features_important, X_input)
    # 2. Hiển thị kết quả: tên biến, important_percentage, actual values
    # dfResult: kết quả từ mô hình: order_code, xác suất và nhãn
    dfResult = dfResult[dfResult['ORDER_CODE'] == order]
    # dataset: dữ liệu raw của các features của mô hình
    dataset = dataset[dataset['ORDER_CODE'] == order]
    dataset.pop('ORDER_CODE')
    dataset = dataset.T
    # 3. Join bảng thứ tự quan trọng với giá trị thực tế
    dataset.columns = ['Features']
    df_important_order = df_weight.join(dataset)

    # 4. Append kết quả vào fraudReason
    fraudReasonList = []
    fraudReasonList = defaultdict()

    fraudReason = df_important_order[['f_weight', 'Features']].reset_index()
    fraudReason = fraudReason[fraudReason['Features'] != 0]
    for idx, item in fraudReason.iterrows():
        try:
            feature_explain = hp.dict_features[item[0]]
        except:
            feature_explain = item[0]
        fraudReasonList[feature_explain] = list(item[1:3].values)

    result = {
        "SOID": order,
        "probability": dfResult['y_prob'].values,
        "predictedClass": dfResult['y_class'].values,
        "modelVersion": "0.0.2",
        "fraudReason": fraudReasonList
        # "fraudReasonSortImportance": details_sort_importance
    }
    return result


def _parse_predict_percentage_important_variables(dfResult, dataset, order, month, df_corr, X_input, modelVersion, df_features_important = None):
    '''
    :param dfResult: Kết quả trả về từ API
    :param dataset: Dữ liệu gốc của đơn hàng
    :param order: Mã đơn hàng
    :param month: Tháng
    :param df_corr: Bảng correction giữa các biến
    :param df_features_important: Bảng features important lấy từ random forest model
    :param X_input: Đầu vào của mô hình
    :return: result of models
    '''
    # 1. Tính mức độ quan trọng important_percentage
    df_weight, order_columns_weight = _cal_weight_percent3(df_corr, X_input)
    # df_weight, order_columns_weight = _cal_weight_percent2(df_corr, df_features_important, X_input)
    # 2. Hiển thị kết quả: tên biến, important_percentage, actual values
    # dfResult: kết quả từ mô hình: order_code, xác suất và nhãn
    dfResult = dfResult[dfResult['ORDER_CODE'] == order]
    # dataset: dữ liệu raw của các features của mô hình
    dataset = dataset[dataset['ORDER_CODE'] == order]
    dataset.pop('ORDER_CODE')
    dataset = dataset.T
    # 3. Join bảng thứ tự quan trọng với giá trị thực tế
    dataset.columns = ['Features']
    df_important_order = df_weight.join(dataset)

    # 4. Append kết quả vào fraudReason
    fraudReasonList = []
    fraudReasonList = defaultdict()

    fraudReason = df_important_order[['f_weight', 'Features']].reset_index()
    fraudReason = fraudReason[fraudReason['Features'] != 0]
    for idx, item in fraudReason.iterrows():
        try:
            feature_explain = _made_feature_into_explaination(item[0])
        except:
            feature_explain = item[0]
        fraudReasonList[feature_explain] = list(item[1:3].values)

    result = {
        "SOID": order,
        "probability": dfResult['y_prob'].values,
        "predictedClass": dfResult['y_class'].values,
        "modelVersion": modelVersion,
        "fraudReason": fraudReasonList
        # "fraudReasonSortImportance": details_sort_importance
    }
    return result

def _made_feature_into_explaination(feature):
    '''
    :param feature: feature name input, features values must be != 0
    :return: explaination from feature name
    :example: "AFF_PARTNERNAME_Zalo" --> "Đối tác đơn hàng Zalo"
    '''
    variable_split = feature.split('_')
    variable_names = ['_'.join(variable_split[:(i+1)]).strip() for i in range(len(variable_split))]
    var_name = [var_name for var_name in variable_names if var_name in hp.dict_variables][0]
    if len(var_name) == len(feature):
        fea_name = ''
    else:
        fea_name = feature[(len(var_name)+1):]
    explaination = ' '.join([hp.dict_variables[var_name], fea_name]).strip()
    return explaination

def _change_type_col(colname, dataset, dtype = 'float', is_replace_na = False):
    if dtype == 'float':
        if is_replace_na:
            dataset[[colname]] = dataset[[colname]].fillna(value = 0)
        dataset[colname] = dataset[colname].apply(lambda x: float(x))
    elif dtype == 'int':
        if is_replace_na:
            dataset[[colname]] = dataset[[colname]].fillna(value = 0)
        dataset[colname] = dataset[colname].apply(lambda x: int(x))
    elif dtype == 'str':
        dataset[colname] = dataset[colname].apply(lambda x: str(x))

def _add_month_filesave(prefix, month):
    file_name = prefix + '_' + str(month) + '.pkl'
    return file_name

def _add_month_version_filesave(prefix, version, month):
    file_name = prefix + '_' + version + '_' + str(month) + '.pkl'
    return file_name


def _query_order_db(cnxn, order_code):
    cnxn = pypyodbc.connect(r"Driver={SQL Server}; Server=ADR-AI\SQLEXPRESS; Database=temp_fc; Trusted_Connection = yes;")


def _updateColumns(dataOrigin, columns_update):
    dataUpdate = pd.DataFrame(np.zeros((dataOrigin.shape[0], len(columns_update))), columns=columns_update)
    dataUpdate.update(dataOrigin)
    return dataUpdate


class NumpyEncoder(json.JSONEncoder):
    '''
    Encoding numpy into json
    '''
    def default(self, obj):
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        if isinstance(obj, np.int32):
            return int(obj)
        if isinstance(obj, np.int64):
            return int(obj)
        if isinstance(obj, np.float32):
            return float(obj)
        if isinstance(obj, np.float64):
            return float(obj)
        return json.JSONEncoder.default(self, obj)


def _process_data_step1(df201906):
    # Ở bước này sẽ đọc dữ liệu từ database trả ra và:
    # 1. Convert dữ liệu về đúng định dạng.
    # 2. Fill các giá trị bị na bằng 0 đối với biến numeric.
    # 3. Thêm các trường DATE, TIME, TIMEUTC, DISCOUNT_AMT (giá trị giảm giá), AFF_EST_COST (chi phí quảng cáo dự kiến)
    # if is_train:
    #     # Nếu dữ liệu lấy từ taining thì thêm 1 trường là Fraud_save_checked
    #     df201906.columns = hp.COLUMNS_QUERY
    # else:
    #     # Nếu dữ liệu là đơn hàng mới thì không thêm trường này
    #     df201906.columns = hp.COLUMNS_QUERY_NEW_ORDER
    # df201906.drop_duplicates('Item_PK', inplace=True)
    # logging.info('df201906 dtypes: {}'.format(df201906[['ORDER_DATE', 'P_QUANTITY', 'AFF_COMMISSION', 'OFFER_FINAL_PRICE',
    #                                           'OFFER_PRICE', 'P_SUBTOTAL', 'P_TOTALPRICE', 'DISCOUNT_AMOUNT',
    #                                           'P_TOTALORDERAMOUNT']].dtypes))
    df201906['ORDER_DATE'] = df201906['ORDER_DATE'].apply(lambda x: datetime.strptime(x, '%Y-%m-%d %H:%M:%S'))
    #logging.info('complated update ORDER_DATE')
    _change_type_col('P_QUANTITY', df201906, is_replace_na=False)
    #logging.info('complated update P_QUANTITY')
    _change_type_col('AFF_COMMISSION', df201906, is_replace_na=True)
    #logging.info('complated update AFF_COMMISSION')
    # _change_type_col('OFFER_FINAL_PRICE', df201906, is_replace_na=False)
    # _change_type_col('OFFER_PRICE', df201906, is_replace_na=False)
    # _change_type_col('P_SUBTOTAL', df201906, is_replace_na=False)
    # _change_type_col('P_TOTALPRICE', df201906, is_replace_na=False)
    # _change_type_col('DISCOUNT_AMOUNT', df201906, is_replace_na=False)
    # _change_type_col('P_TOTALORDERAMOUNT', df201906, is_replace_na=False)
    # Thêm biến về DATE và TIME tạo đơn hàng
    df201906['DATE'] = df201906['ORDER_DATE'].apply(lambda x: x.date())
    df201906['TIME'] = df201906['ORDER_DATE'].apply(lambda x: x.time())
    df201906['TIMEUTC'] = df201906['ORDER_DATE'].apply(lambda x: x.timestamp() / 86400)
    #logging.info('complated create DATE, TIME, TIMEUTC')
    # Update lại DISCOUNT_AMOUNT
    df201906['DISCOUNT_AMOUNT'] = df201906['OFFER_PRICE'] - df201906['OFFER_FINAL_PRICE']
    #logging.info('complated update DISCOUNT_AMOUNT')
    # Tính chi phí quảng cáo dự kiến đối với từng ORDER_CODE
    df201906['AFF_EST_COST'] = df201906['OFFER_FINAL_PRICE'] * df201906['AFF_COMMISSION'] / 100
    #logging.info('complated update AFF_EST_COST')
    return df201906

def _predict_prob_class(model, X, thres = 0.5):
    y_prob = model.predict_proba(X)
    if (y_prob.ndim == 2) and (y_prob.shape[1] == 2):
        y_prob = y_prob[:, 1]
        y_class = [0 if prob < thres else 1 for prob in y_prob]
    elif (y_prob.ndim == 2) and (y_prob.shape[1] == 1):
        y_prob = y_prob[:, 0]
        y_class = [0 if prob < thres else 1 for prob in y_prob]
    elif y_prob.ndim == 1:
        y_class = [0 if prob < thres else 1 for prob in y_prob]
    return y_prob, y_class


def _rename_col_dfsumary(dict_cols, columns):
    return [dict_cols[key] for key in columns]

def _gc_collect(table_name):
    del table_name
    gc.collect()

_count_unique = lambda x: len(x.unique())

def _dfOrder(df201906, is_train = False, order_id = None):
    if is_train:
        dfOrder201906 = pd.pivot_table(df201906,
                                       values=hp.ITEM_COLS_MEA + hp.ORDER_COLS_MEA + ['OFFER_CODE', 'CODE2_MAIN', 'CODE5_MAIN'],
                                       index='ORDER_CODE',
                                       aggfunc={
                                           'P_QUANTITY': np.sum,  # Tổng số lượng item trong đơn hàng
                                           'AFF_COMMISSION': np.mean,  # Trung bình cộng phí quảng cáo
                                           'OFFER_FINAL_PRICE': np.mean,  # Trung bình giá trị của 1 Item sau khuyến mại
                                           'OFFER_PRICE': np.mean,  # Trung bình giá trị gốc của 1 Item
                                           'DISCOUNT_AMOUNT': np.sum,  # Tổng giá trị được giảm giá của đơn hàng
                                           'AFF_EST_COST': np.sum,  # Tổng chi phí quảng cáo đơn hàng
                                           'P_SUBTOTAL': np.mean,  # Tổng chi phí đơn hàng
                                           'P_TOTALPRICE': np.mean,
                                           # Tổng giá trị phải thanh toán đơn hàng sau khi trừ đi chiết khấu Vin
                                           'P_TOTALORDERAMOUNT': np.mean,
                                           # Tổng giá trị phải thanh toán đơn hàng trước khi trừ đi chiết khấu Vin
                                           'OFFER_CODE': _count_unique,  # Số lượng OFFER_CODE
                                           'CODE2_MAIN': _count_unique,  # Số lượng MAIN CODE level 2
                                           'CODE5_MAIN': _count_unique  # Số lượng MAIN CODE level 5
                                       }).reset_index()
    else:
        dfOrder201906 = pd.pivot_table(df201906[df201906['ORDER_CODE'] == order_id],
                                       values=hp.ITEM_COLS_MEA + hp.ORDER_COLS_MEA + ['OFFER_CODE', 'CODE2_MAIN', 'CODE5_MAIN'],
                                       index='ORDER_CODE',
                                       aggfunc={
                                           'P_QUANTITY': np.sum,  # Tổng số lượng item trong đơn hàng
                                           'AFF_COMMISSION': np.mean,  # Trung bình cộng phí quảng cáo
                                           'OFFER_FINAL_PRICE': np.mean,  # Trung bình giá trị của 1 Item sau khuyến mại
                                           'OFFER_PRICE': np.mean,  # Trung bình giá trị gốc của 1 Item
                                           'DISCOUNT_AMOUNT': np.sum,  # Tổng giá trị được giảm giá của đơn hàng
                                           'AFF_EST_COST': np.sum,  # Tổng chi phí quảng cáo đơn hàng
                                           'P_SUBTOTAL': np.mean,  # Tổng chi phí đơn hàng
                                           'P_TOTALPRICE': np.mean,
                                           # Tổng giá trị phải thanh toán đơn hàng sau khi trừ đi chiết khấu Vin
                                           'P_TOTALORDERAMOUNT': np.mean,
                                           # Tổng giá trị phải thanh toán đơn hàng trước khi trừ đi chiết khấu Vin
                                           'OFFER_CODE': _count_unique, # Số lượng OFFER_CODE
                                           'CODE2_MAIN': _count_unique, # Số lượng MAIN CODE level 2
                                           'CODE5_MAIN': _count_unique # Số lượng MAIN CODE level 5
                                       }).reset_index()
    return dfOrder201906

def _dfIPQuantity(df201906, is_train = False, ip = None):
    if is_train:
        dfIPQuantity = pd.pivot_table(df201906,
                                      values=['CODE2_MAIN', 'CODE5_MAIN', 'EMAIL_ACCOUNT', 'OFFER_CODE',
                                              'PHONE_DELIVERY',
                                              'ORDER_VINIDNUMBER', 'TIMEUTC', 'WARD'],
                                      index=['IP_ADDRESS'],
                                      aggfunc={
                                          'CODE2_MAIN': _count_unique,
                                          'CODE5_MAIN': _count_unique,
                                          'EMAIL_ACCOUNT': _count_unique,
                                          'OFFER_CODE': _count_unique,
                                          'ORDER_VINIDNUMBER': _count_unique,
                                          'PHONE_DELIVERY': _count_unique,
                                          'TIMEUTC': np.std,
                                          'WARD': _count_unique
                                      }).reset_index()
    else:
        dfIPQuantity = pd.pivot_table(df201906[df201906['IP_ADDRESS'] == ip],
                                      values=['CODE2_MAIN', 'CODE5_MAIN', 'EMAIL_ACCOUNT', 'OFFER_CODE', 'PHONE_DELIVERY',
                                              'ORDER_VINIDNUMBER', 'TIMEUTC', 'WARD'],
                                      index=['IP_ADDRESS'],
                                      aggfunc={
                                          'CODE2_MAIN': _count_unique,
                                          'CODE5_MAIN': _count_unique,
                                          'EMAIL_ACCOUNT': _count_unique,
                                          'OFFER_CODE': _count_unique,
                                          'ORDER_VINIDNUMBER': _count_unique,
                                          'PHONE_DELIVERY': _count_unique,
                                          'TIMEUTC': np.std,
                                          'WARD': _count_unique
                                      }).reset_index()
    return dfIPQuantity

def _dfPhoneQuantity(df201906, is_train = False, phone = None):
    if is_train:
        dfPhoneQuantity = pd.pivot_table(df201906,
                                         values=['CODE2_MAIN', 'CODE5_MAIN', 'EMAIL_ACCOUNT', 'OFFER_CODE',
                                                 'IP_ADDRESS',
                                                 'ORDER_VINIDNUMBER', 'TIMEUTC', 'WARD'],
                                         index=['PHONE_DELIVERY'],
                                         aggfunc={
                                             'CODE2_MAIN': _count_unique,
                                             'CODE5_MAIN': _count_unique,
                                             'EMAIL_ACCOUNT': _count_unique,
                                             'OFFER_CODE': _count_unique,
                                             'ORDER_VINIDNUMBER': _count_unique,
                                             'IP_ADDRESS': _count_unique,
                                             'TIMEUTC': np.std,
                                             'WARD': _count_unique
                                         }).reset_index()
    else:
        dfPhoneQuantity = pd.pivot_table(df201906[df201906['PHONE_DELIVERY'] == phone],
                                         values=['CODE2_MAIN', 'CODE5_MAIN', 'EMAIL_ACCOUNT', 'OFFER_CODE', 'IP_ADDRESS',
                                                 'ORDER_VINIDNUMBER', 'TIMEUTC', 'WARD'],
                                         index=['PHONE_DELIVERY'],
                                         aggfunc={
                                             'CODE2_MAIN': _count_unique,
                                             'CODE5_MAIN': _count_unique,
                                             'EMAIL_ACCOUNT': _count_unique,
                                             'OFFER_CODE': _count_unique,
                                             'ORDER_VINIDNUMBER': _count_unique,
                                             'IP_ADDRESS': _count_unique,
                                             'TIMEUTC': np.std,
                                             'WARD': _count_unique
                                         }).reset_index()
    return dfPhoneQuantity

def _dfVinIDQuantity(df201906, is_train = False, acc = None):
    if is_train:
        dfVinIDQuantity = pd.pivot_table(df201906,
                                         values=['CODE2_MAIN', 'CODE5_MAIN', 'EMAIL_ACCOUNT', 'OFFER_CODE',
                                                 'IP_ADDRESS',
                                                 'PHONE_DELIVERY', 'TIMEUTC', 'WARD'],
                                         index=['ORDER_VINIDNUMBER'],
                                         aggfunc={
                                             'CODE2_MAIN': _count_unique,
                                             'CODE5_MAIN': _count_unique,
                                             'EMAIL_ACCOUNT': _count_unique,
                                             'OFFER_CODE': _count_unique,
                                             'IP_ADDRESS': _count_unique,
                                             'PHONE_DELIVERY': _count_unique,
                                             'TIMEUTC': np.std,
                                             'WARD': _count_unique
                                         }).reset_index()
    else:
        dfVinIDQuantity = pd.pivot_table(df201906[df201906['ORDER_VINIDNUMBER'] == acc],
                                         values=['CODE2_MAIN', 'CODE5_MAIN', 'EMAIL_ACCOUNT', 'OFFER_CODE',
                                                 'IP_ADDRESS',
                                                 'PHONE_DELIVERY', 'TIMEUTC', 'WARD'],
                                         index=['ORDER_VINIDNUMBER'],
                                         aggfunc={
                                             'CODE2_MAIN': _count_unique,
                                             'CODE5_MAIN': _count_unique,
                                             'EMAIL_ACCOUNT': _count_unique,
                                             'OFFER_CODE': _count_unique,
                                             'IP_ADDRESS': _count_unique,
                                             'PHONE_DELIVERY': _count_unique,
                                             'TIMEUTC': np.std,
                                             'WARD': _count_unique
                                         }).reset_index()
    return dfVinIDQuantity

def _correlation(X_train, y_train, final_features):
    coefs = []
    for i in np.arange(X_train.shape[1]):
        coef = np.corrcoef(X_train[:, i], y_train)[0, 1]
        if np.isnan(coef):
            coef = 0
        coefs.append(coef)
    coefs = np.array(coefs)
    df_corr = pd.DataFrame({'Correlation': coefs}, index=final_features)
    return df_corr

def _cal_weight_percent(df_corr, df_final_features, X_input):
    '''
    :Formula: W(exp(X*C))*F (X - values, C - correlation, F - features important coefs)
    :param df_corr: correlation
    :param df_final_features:  important coef from random forest
    :param X_input: X scaled values
    :return: weight and columns ascend sorting by weight
    '''
    df_weight = df_corr.join(df_final_features)
    df_weight['X'] = X_input
    df_weight['exp'] = np.exp(df_weight['X'] * df_weight['Correlation'])
    df_weight['f_new'] = df_weight['importance values'] * df_weight['exp']
    df_weight['f_weight'] = df_weight['f_new'] / np.sum(df_weight['f_new'])
    df_weight = df_weight.sort_values('f_weight', ascending=False)
    order_columns_weight = df_weight.index
    return df_weight, order_columns_weight

def _cal_weight_percent2(df_corr, df_features_important, X_input):
    '''
    :Formula: X*C*F (X - values, C - correlation, F - features important coefs)
    :param df_corr: correlation
    :param df_features_important:  important coef from random forest
    :param X_input: X scaled values
    :return: weight and columns ascend sorting by weight
    '''
    df_weight = df_corr.join(df_features_important)
    df_weight['X'] = X_input
    df_weight['f_weight'] = df_weight['X'] * df_weight['Correlation'] * np.log(df_weight['importance values'])
    df_weight['f_weight'] = df_weight['f_weight']/np.sum(df_weight['f_weight'])
    df_weight = df_weight.sort_values('f_weight', ascending=False)
    order_columns_weight = df_weight.index
    return df_weight, order_columns_weight

def _cal_weight_percent3(df_corr, X_input, df_final_features = None):
    '''
    :Formula: X*C (X - values, C - correlation)
    :param df_corr: correlation
    :param df_final_features:  important coef from random forest
    :param X_input: X scaled values
    :return: weight and columns ascend sorting by weight
    '''
    df_corr['X'] = X_input
    df_corr['f_weight'] = df_corr['X'] * df_corr['Correlation']
    df_corr['f_weight'] = df_corr['f_weight']/np.sum(df_corr['f_weight'])
    df_corr = df_corr.sort_values('f_weight', ascending=False)
    order_columns_weight = df_corr.index
    return df_corr, order_columns_weight
