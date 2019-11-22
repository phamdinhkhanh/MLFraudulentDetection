import pandas as pd
import numpy as np
import string
import logging
import time
import pypyodbc
import gc
import utils
import hyperameter as hp
from datetime import datetime
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from datetime import datetime, timedelta
logging.basicConfig(filename='persistentvolume/log_{}.log'.format(datetime.strftime(datetime.now(), '%Y%m%d')),
                    filemode='a',format='%(asctime)s: %(levelname)s : %(message)s',
                    level = logging.INFO)

def _read_data(is_train = True, order_id = None):
    # Khởi tạo các trường ip, acc, phone is None
    ip = None
    phone = None
    acc = None
    # I. Trường hợp dữ liệu training
    if is_train:
        start_time = datetime.now()
        logging.info('month: {}'.format(hp.month))
        # Lấy ra danh sách các đơn hàng theo month
        df201906 = utils._getAllOrder1MTraining(hp.previous_30_days_train, hp.firstDayThisMonth)
        logging.info('previous 30 days training: {}'.format(hp.previous_30_days_train))
        logging.info('first day of this month: {}'.format(hp.firstDayThisMonth))
        logging.info('df201906 shape: {}'.format(df201906.shape))
        logging.info('df201906 dtypes: {}'.format(df201906.dtypes))
        df201906 = utils._process_data_step1(df201906)
        end_time = datetime.now()
        logging.info('0. Time execution on load all orders in month: {}, df201906 shape: {}'.format(
            end_time - start_time,
            df201906.shape))
    else:
    # II. Trường hợp dữ liệu dùng để dự báo đơn hàng trong ngày
        # là dữ liệu đơn hàng phát sinh trong ngày.
        # Từ order_id truyền vào lấy ra IP, PHONE, VINIDACCOUNT
        if order_id is None:
            raise Exception('Value of order_id is None')

        start_time = datetime.now()
        # sqlqueryOrderId = utils._sql_query_OrderID(hp.TABLE_REALTIME, order_id)
        # dfOrder = pd.read_sql_query(sqlqueryOrderId, hp.cnxn).drop_duplicates()
        # \\MODIFIED: Thay query tới local server bằng procedure View_Get_P_I_V_Info_ByOrder (A.Thuận)
        dfOrder = utils._getInfoByOrder(p_orderid=order_id)
        if dfOrder.shape[0] == 0:
            raise Exception('orderId not found in Database')

        phone = dfOrder['PHONE_DELIVERY'].tolist()[0]
        ip = dfOrder['IP_ADDRESS'].tolist()[0]
        acc = dfOrder['ORDER_VINIDNUMBER'].tolist()[0]

        logging.info('order_id: {}'.format(order_id))
        logging.info('phone: {}'.format(phone))
        logging.info('ip: {}'.format(ip))
        logging.info('acc: {}'.format(acc))
        end_time = datetime.now()
        logging.info('1. time of loading data: SQL select IP, PHONE, VINACC from ORDER_CODE: {}'.format(end_time-start_time))
        # Load dữ liệu trong tháng giả sử nó đã tồn tại trong caches
        # Option 2: Đọc dữ liệu trực tiếp từ SQL
        start_time = datetime.now()
        # Lấy các trường 1M theo điều kiện phone, ip, acc
        df201906 = utils._getAllOrder1MByPhoneIpAcc(p_phone = phone, p_ip=ip, p_acc=acc)
        end_time = datetime.now()
        logging.info('1. time of loading data: query past 30 days by conditions PHONE, IP, VINACC: {}, df201906 shape: {}'.format(end_time - start_time, df201906.shape))
        start_time = datetime.now()
        df201906 = utils._process_data_step1(df201906)
        end_time = datetime.now()
        logging.info('1. time of loading data: processing data step 1 past 30 days by conditions PHONE, IP, VINACC: {}, df201906 shape: {}'.format(end_time - start_time, df201906.shape))
        # Load dữ liệu các đơn hàng mới phát sinh dựa trên phone, ip, acc
        # Note: bỏ Fraud_check vì các đơn hàng chưa có dữ liệu fraud check
        start_time = datetime.now()
        # \\MODIFIED: Thay query tới local server bằng procedure View_GetAllInfo_ByOrder_Today (A.Thuận)
        # sqlqueryNewOrder = utils._sql_query_new_order(hp.TABLE_REALTIME, phone = phone, ip = ip, acc = acc)
        # dfNewData = pd.read_sql_query(sqlqueryNewOrder, hp.cnxn).drop_duplicates()
        dfNewData = utils._getAllOrderTodayByOrder(p_orderid=order_id)
        if dfNewData.shape[0] == 0:
            raise Exception('orderID not found in Database')
        # Đi qua preproccessing data step 1 để chuẩn hóa dữ liệu và tạo một số trường mới như: DATE, TIME, TIMEUTC, DISCOUNT_AMT (giá trị giảm giá), AFF_EST_COST (chi phí quảng cáo dự kiến)
        dfNewData = utils._process_data_step1(dfNewData)
        end_time = datetime.now()
        logging.info('1. time of loading data: SQL select today ORDER_ID: {}'.format(end_time - start_time))
        assert dfNewData.shape[1] == df201906.shape[1]
        start_time = datetime.now()
        df201906 = pd.concat([df201906, dfNewData], axis = 0)
        # delete dfNewData
        utils._gc_collect(dfNewData)
        df201906.index = np.arange(df201906.shape[0])
        logging.info('1. time of loading data: shape of concatenate dfNewData and df201906: {}'.format(df201906.shape))
        end_time = datetime.now()
        logging.info('1. time of loading data: time concatenate today order with 30 days history: {}'.format(end_time - start_time))

    # Nhóm biến theo đơn hàng
    order_cols_mea = hp.ORDER_COLS_MEA
    order_cols_dim_train = hp.ORDER_COLS_DIM_TRAIN
    order_cols_dim_new = hp.ORDER_COLS_DIM_NEW

    # Nhóm biến theo ITEM_KEY
    item_cols_mea = hp.ITEM_COLS_MEA
    start_time = datetime.now()
    _count_unique = lambda x: len(x.unique())

    # Đếm số lượng mặt hàng và số lượng các ngành hàng theo CODE2_MAIN, CODE5_MAIN, OFFER_CODE theo ORDER_CODE
    # Tính giá trị đơn hàng, chi phí, giá trị commission số lượng hàng
    # Tính chi phí quảng cáo trung bình của đơn hàng theo AFF_EST_COST

    dfOrder201906 = utils._dfOrder(df201906, is_train, order_id)
    # Tính tỷ lệ chi phí quảng cáo/tổng giá trị đơn hàng.
    dfOrder201906['AFF_COMMISSION_FEE'] = dfOrder201906['AFF_EST_COST'] / dfOrder201906['P_TOTALORDERAMOUNT'] * 100
    dfOrder201906.loc[np.where(np.isinf(dfOrder201906['AFF_COMMISSION_FEE']))[0], 'AFF_COMMISSION_FEE'] = 0
    if is_train:
        dfOrderDim = df201906[order_cols_dim_train].drop_duplicates()
    else:
        dfOrderDim = df201906[df201906['ORDER_CODE'] == order_id][order_cols_dim_new].drop_duplicates()

    assert dfOrderDim.shape[0] == dfOrder201906.shape[0]
    # Join dữ liệu OrderDim với dữ liệu measurement dfOrder201906
    columns = ['ORDER_CODE', 'AFF_COMMISSION', 'AFF_EST_COST', 'N_CODE2_MAIN',
                'N_CODE5_MAIN', 'DISCOUNT_AMOUNT', 'N_OFFER_CODE', 'OFFER_FINAL_PRICE',
                'OFFER_PRICE', 'P_QUANTITY', 'P_SUBTOTAL', 'P_TOTALORDERAMOUNT',
                'P_TOTALPRICE', 'AFF_COMMISSION_FEE']

    dfTrainSum201906 = utils._merge_summary_dimension(dfOrder201906, dfOrderDim,
                                                      columns=columns, join_column='ORDER_CODE')
    end_time = datetime.now()
    logging.info('1. time of loading data: Time summary Order: {}'.format(end_time-start_time))
    # delete dfOrderDim & dfOrder201906
    utils._gc_collect(dfOrderDim)
    utils._gc_collect(dfOrder201906)

    # Đếm số lượng mặt hàng và số lượng các ngành hàng và số lượng PHONE sử dụng theo IP_ADDRESS
    # CODE2_MAIN, CODE5_MAIN, EMAIL_ACCOUNT, OFFER_CODE, PHONE_DELIVERY, ORDER_VINIDNUMBER,  WARD, TIMEUTC

    dfIPQuantity = utils._dfIPQuantity(df201906, is_train, ip)
    dict_columns = {
        'IP_ADDRESS':'IP_ADDRESS',
        'CODE2_MAIN': 'IP_N_CODE2_MAIN',
        'CODE5_MAIN': 'IP_N_CODE5_MAIN',
        'EMAIL_ACCOUNT': 'IP_N_EMAIL_ACCOUNT',
        'OFFER_CODE': 'IP_N_OFFER_CODE',
        'ORDER_VINIDNUMBER': 'IP_N_ORDER_VINACCOUNT',
        'PHONE_DELIVERY': 'IP_N_PHONE',
        'TIMEUTC': 'IP_STD_TIMEUTC',
        'WARD': 'IP_N_WARD'
    }
    columns = utils._rename_col_dfsumary(dict_columns, dfIPQuantity.columns)
    dfTrainSum201906 = utils._merge_summary_dimension(dfIPQuantity, dfTrainSum201906,
                                                      columns=columns, join_column='IP_ADDRESS')

    # Đếm số lượng mặt hàng và số lượng các ngành hàng số lượng IP sử dụng theo PHONE
    # CODE2_MAIN, CODE5_MAIN, EMAIL_ACCOUNT, OFFER_CODE, PHONE_DELIVERY, ORDER_VINIDNUMBER, WARD, TIMEUTC
    end_time = datetime.now()
    logging.info('1. time of loading data: Time count IP Order: {}'.format(end_time - start_time))
    # delete dfIPQuantity
    utils._gc_collect(dfIPQuantity)
    start_time = datetime.now()

    dfPhoneQuantity = utils._dfPhoneQuantity(df201906, is_train, phone)
    dict_columns = {
        'PHONE_DELIVERY':'PHONE_DELIVERY',
        'CODE2_MAIN':'PHONE_N_CODE2_MAIN',
        'CODE5_MAIN':'PHONE_N_CODE5_MAIN',
        'EMAIL_ACCOUNT':'PHONE_N_EMAIL_ACCOUNT',
        'OFFER_CODE':'PHONE_N_OFFER_CODE',
        'ORDER_VINIDNUMBER':'PHONE_N_ORDER_VINACCOUNT',
        'IP_ADDRESS':'PHONE_N_IP_ADDRESS',
        'TIMEUTC':'PHONE_STD_TIMEUTC',
        'WARD':'PHONE_N_WARD'
    }
    columns = utils._rename_col_dfsumary(dict_columns, dfPhoneQuantity.columns)
    dfTrainSum201906 = utils._merge_summary_dimension(dfPhoneQuantity, dfTrainSum201906,
                                                      columns=columns, join_column='PHONE_DELIVERY')
    end_time = datetime.now()
    logging.info('1. time of loading data: Time count Phone Order: {}'.format(end_time - start_time))
    # delete dfPhoneQuantity
    utils._gc_collect(dfPhoneQuantity)
    # Đếm số lượng mặt hàng và số lượng các ngành hàng theo ORDER_VINIDNUMBER
    # CODE2_MAIN, CODE5_MAIN, EMAIL_ACCOUNT, OFFER_CODE, PHONE_DELIVERY, ORDER_VINIDNUMBER, WARD, TIMEUTC
    start_time = datetime.now()

    dfVinIDQuantity = utils._dfVinIDQuantity(df201906, is_train, acc)
    dict_columns = {
        'ORDER_VINIDNUMBER':'ORDER_VINIDNUMBER',
        'CODE2_MAIN': 'ORDER_VINACC_N_CODE2_MAIN',
        'CODE5_MAIN': 'ORDER_VINACC_N_CODE5_MAIN',
        'EMAIL_ACCOUNT': 'ORDER_VINACC_N_EMAIL_ACCOUNT',
        'OFFER_CODE': 'ORDER_VINACC_N_OFFER_CODE',
        'IP_ADDRESS': 'ORDER_VINACC_N_IP_ADDRESS',
        'PHONE_DELIVERY': 'ORDER_VINACC_N_PHONE',
        'TIMEUTC': 'ORDER_VINACC_STD_TIMEUTC',
        'WARD': 'ORDER_VINACC_N_WARD'
    }
    columns = utils._rename_col_dfsumary(dict_columns, dfVinIDQuantity.columns)
    dfTrainSum201906 = utils._merge_summary_dimension(dfVinIDQuantity, dfTrainSum201906,
                                                      columns=columns, join_column='ORDER_VINIDNUMBER')
    end_time = datetime.now()
    logging.info('1. time of loading data: Time count VinID Order: {}'.format(end_time - start_time))
    # delete dfVinIDQuantity
    utils._gc_collect(dfVinIDQuantity)
    # Thống kê số lượng đơn hàng đặt theo IP trong 30 ngày
    start_time = datetime.now()
    if is_train:
        df_ip = pd.pivot_table(df201906[['ORDER_CODE', 'IP_ADDRESS', 'DATE']],
                               index=['IP_ADDRESS'],
                               columns=['DATE'],
                               aggfunc={'ORDER_CODE': len})
    else:
        df_ip = pd.pivot_table(df201906[df201906['IP_ADDRESS'] == ip][['ORDER_CODE', 'IP_ADDRESS', 'DATE']],
                               index=['IP_ADDRESS'],
                               columns=['DATE'],
                               aggfunc={'ORDER_CODE': len})
    if df_ip.empty:
        logging.info('1. time of loading data: df_ip is empty!')
        date_range = pd.date_range(hp.previous_30_days, hp.today)
        df_ip = pd.DataFrame(np.zeros((df_ip.shape[0], len(date_range))), columns=date_range)
    else:
        df_ip.columns = [datetime.strftime(x, '%Y-%m-%d') for x in df_ip.columns.levels[1]]
        date_range = pd.date_range(hp.previous_30_days, hp.today)
        df_ip_full_days = pd.DataFrame(np.zeros((df_ip.shape[0], len(date_range))), columns=date_range)
        df_ip_full_days.update(df_ip)
        df_ip = df_ip_full_days

    # Đánh dấu IP đặt hàng trong vòng 30 ngày có số lượng đơn hàng 2, 5, 10, 20
    df_summary = df_ip.apply(np.nansum, axis=1)

    IP_SUSPECT_30_2 = list(df_summary[df_summary >= 2].index)
    IP_SUSPECT_30_5 = list(df_summary[df_summary >= 5].index)
    IP_SUSPECT_30_10 = list(df_summary[df_summary >= 10].index)
    IP_SUSPECT_30_20 = list(df_summary[df_summary >= 20].index)

    # delete df_summary
    utils._gc_collect(df_summary)

    dfTrainSum201906['IP_SUSPECT_30_2'] = [1 if ip in IP_SUSPECT_30_2 else 0 for ip in dfTrainSum201906['IP_ADDRESS']]
    dfTrainSum201906['IP_SUSPECT_30_5'] = [1 if ip in IP_SUSPECT_30_5 else 0 for ip in dfTrainSum201906['IP_ADDRESS']]
    dfTrainSum201906['IP_SUSPECT_30_10'] = [1 if ip in IP_SUSPECT_30_10 else 0 for ip in dfTrainSum201906['IP_ADDRESS']]
    dfTrainSum201906['IP_SUSPECT_30_20'] = [1 if ip in IP_SUSPECT_30_20 else 0 for ip in dfTrainSum201906['IP_ADDRESS']]

    # Đánh dấu IP có số lượng đơn hàng đặt trong tuần >= 5
    dict_ip = {
        "IP_SUSPECT_WEEK0": None,
        "IP_SUSPECT_WEEK1": None,
        "IP_SUSPECT_WEEK2": None,
        "IP_SUSPECT_WEEK3": None,
        "IP_SUSPECT_WEEK4": None,
        "IP_SUSPECT_WEEK5": None,
        "IP_SUSPECT_WEEK6": None
    }

    for i in range(int(df_ip.shape[1] / 7)):
        df_summary_week = df_ip.iloc[:, (7 * i):(7 * (i + 1))].apply(np.nansum, axis=1)
        element = "IP_SUSPECT_WEEK" + str(i)
        dict_ip[element] = list(df_summary_week[df_summary_week >= 5].index)
        dfTrainSum201906[element] = [1 if ip in dict_ip[element] else 0 for ip in dfTrainSum201906['IP_ADDRESS']]


    # Đếm số lượng tuần có giao dịch >= 5 ('IP_SUSPECT_WEEK_NUM') và đánh dấu IP nếu tồn tại tuần trong tháng có giao dịch nhiều hơn >= 5 ('IP_SUSPECT_WEEK')
    filter_cols = [colname for colname in dfTrainSum201906.columns if colname.startswith('IP_SUSPECT_WEEK')]
    dfTrainSum201906['IP_SUSPECT_WEEK_NUM'] = dfTrainSum201906[filter_cols].apply(sum, axis=1)
    dfTrainSum201906['IP_SUSPECT_WEEK'] = [1 if week_susp > 0 else 0 for week_susp in
                                           dfTrainSum201906['IP_SUSPECT_WEEK_NUM']]

    end_time = datetime.now()
    logging.info('1. time of loading data: Time update transaction IP: {}'.format(end_time - start_time))
    # Xử lý PAYMENTMODES
    def _group_payment_mode(paymode):
        if paymode is None:
            return None
        elif paymode[:8] == 'DISCOUNT':
            return 'DISCOUNT'
        elif paymode[:7] == 'Tra gop':
            return 'TRA_GOP'
        elif paymode[:7] == 'INVOICE':
            return 'INVOICE'
        elif 'NAPAS' in paymode:
            return 'NAPAS'
        elif (paymode[:7] == 'EWallet'):
            return 'EWALLET'
        elif 'ONEPAY' in paymode:
            return 'ONEPAY'
        else:
            return paymode

    start_time = datetime.now()
    # Tạo trường PAYMENTMODES_GROUP
    dfTrainSum201906['PAYMENTMODES_GROUP'] = dfTrainSum201906['PAYMENTMODES'].apply(_group_payment_mode)

    # Thông tin về thẻ ORDER_USER_VINIDACCOUNT và ORDER_VINIDNUMBER có trùng nhau hay không
    dfTrainSum201906['IS_VINID_MATCH'] = dfTrainSum201906['ORDER_USER_VINIDACCOUNT'] == dfTrainSum201906['ORDER_VINIDNUMBER']
    dfTrainSum201906['IS_VINID_MATCH'] = [1 if item else 0 for item in dfTrainSum201906['IS_VINID_MATCH']]

    # Tạo trường sử dụng IS_AFF_USE
    dfTrainSum201906['IS_AFF_USE'] = [1 if item > 0 else 0 for item in dfTrainSum201906['AFF_EST_COST']]

    # Cập nhật AFF_SOURCE: Đối với 1 đơn hàng có nhiều AFFILIATE_PARTNERNAME thì chỉ lấy AFFILIATE_PARTNERNAME có chi phí quảng cáo là lớn nhất để update cho toàn bộ đơn hàng.
    dfAff = pd.DataFrame(df201906[['ORDER_CODE', 'AFF_PARTNERNAME', 'AFF_EST_COST']].groupby(
        ['ORDER_CODE', 'AFF_PARTNERNAME']).AFF_EST_COST.sum())
    dfAff = dfAff.reset_index().sort_values(by='AFF_EST_COST')[['ORDER_CODE', 'AFF_PARTNERNAME']]
    dfAff.drop_duplicates(inplace=True)
    dfTrainSum201906 = pd.merge(dfTrainSum201906, dfAff, left_on='ORDER_CODE', right_on='ORDER_CODE', how='left')

    # So khớp EMAIL_ACCONT và EMAIL_DELIVERY
    dfTrainSum201906['EMAIL_ACCOUNT'] = dfTrainSum201906['EMAIL_ACCOUNT'].str.lower()
    dfTrainSum201906['EMAIL_DELIVERY'] = dfTrainSum201906['EMAIL_DELIVERY'].str.lower()
    IS_MATCH_EMAIL = dfTrainSum201906['EMAIL_ACCOUNT'] == dfTrainSum201906['EMAIL_DELIVERY']
    IS_MATCH_EMAIL = [1 if item else 0 for item in IS_MATCH_EMAIL]
    dfTrainSum201906['IS_MATCH_EMAIL'] = IS_MATCH_EMAIL

    # Check email adayroi
    def _check_adr_mail(email):
        if email is None:
            return 0
        elif 'adayroi' in email:
            return 1
        else:
            return 0
    dfTrainSum201906['IS_ADR_EMAIL_ACCOUNT'] = dfTrainSum201906['EMAIL_ACCOUNT'].apply(_check_adr_mail)
    dfTrainSum201906['IS_ADR_EMAIL_DELIVERY'] = dfTrainSum201906['EMAIL_DELIVERY'].apply(_check_adr_mail)
    end_time = datetime.now()
    logging.info('1. time of loading data: Time Update these other (email, affpartner, affsource): {}'.format(end_time - start_time))
    if is_train:
        col_selected = hp.COL_SELECTED_TRAIN
    else:
        col_selected = hp.COL_SELECTED_NEW_ORDER

    start_time = datetime.now()
    # Update những features mà trên dataset train không có [IP_WEEK_1, IP_WEEK_2,..., IP_WEEK_3]
    dfTrainSel201906 = utils._updateColumns(dfTrainSum201906, col_selected)
    # logging.info('dfTrainSel201906 {} columns: {}'.format(len(dfTrainSel201906.columns), dfTrainSel201906.columns))
    dfTrainSelFullCol201906_dummy = pd.get_dummies(dfTrainSel201906,
                                      columns=['PAYMENTMODES_GROUP', 'PAYMENTSTATUS',
                                               'SALES_APPLICATION', 'AFF_PARTNERNAME'])

    if not is_train:
        final_features_fn = utils._add_month_filesave(hp.FINAL_FEATURES, hp.month)
        final_features = utils.IOObject(final_features_fn)._load_pickle()
        #final_features = ['ORDER_CODE'] + final_features
        logging.info('final_features_fn folder name: {}'.format(final_features_fn))
        logging.info('final_features columns of dfTrainSelFullCol201906_dummy: {}'.format(final_features))
        dfTrainSelFullCol201906_dummy = utils._updateColumns(dfTrainSelFullCol201906_dummy, columns_update=final_features)

    ORDER_CODE = dfTrainSelFullCol201906_dummy.pop('ORDER_CODE')
    utils._gc_collect(dfTrainSum201906)
    if is_train:
        Target = dfTrainSelFullCol201906_dummy.pop('FRAUD_SaveValue_Check')
        Target = [1 if check == 'YES' else 0 for check in Target]
        scaler = MinMaxScaler(feature_range=(-1, 1))
        scaler.fit(dfTrainSelFullCol201906_dummy)
        X = scaler.transform(dfTrainSelFullCol201906_dummy)
        y = np.array(Target)
        X_train, X_test, y_train, y_test, id_train, id_test = train_test_split(X, y, np.arange(dfTrainSelFullCol201906_dummy.shape[0]), \
                                                                               test_size=0.2, stratify=Target, random_state=123)
        # delete dfTrainSel201906
        utils._gc_collect(dfTrainSel201906)
        dfTrainSelFullCol201906_dummy['ORDER_CODE'] = ORDER_CODE
        return dfTrainSelFullCol201906_dummy, X_train, X_test, y_train, y_test, scaler
    else:
        # Load scaler từ catches
        try:
            scaler_fn = utils._add_month_filesave(hp.SCALER_FILE, hp.month)
            scaler = utils.IOObject(scaler_fn)._load_pickle()
        except:
            raise Exception('There are not any scaler!')
        X = scaler.transform(dfTrainSelFullCol201906_dummy)
        end_time = datetime.now()
        dfTrainSelFullCol201906_dummy['ORDER_CODE'] = ORDER_CODE
        logging.info('1. time of loading data: Time get_dummy and create X input: {}'.format(end_time - start_time))
        return ORDER_CODE, dfTrainSelFullCol201906_dummy, X