from datetime import datetime, timedelta
import cx_Oracle
import pypyodbc
import json
pypyodbc.lowercase = False

with open('config.json', "r") as fp:
    config = json.loads(fp.read())
print('config: {}'.format(config))

# ================================Phần tham số config=====================================
# \\TODO: Thay IP bằng domain khi lên production
IP = config['ip']
# DONE \\TODO: Thay cnxn khi lên production
# cnxn = cx_Oracle.connect(config['connect_string'])
constring = config['connect_string']
# DONE
TABLE_30DAYS = config['table_name_30_days']
# \\TODO: Thay bằng tên bảng lưu trữ 30 days khi lên production
# TABLE_REALTIME = "TMP_FCKB_RealTime"
# DONE \\TODO: Thay bằng giá trị của ngày today khi lên production
# today = config['today']
today = datetime.strftime(datetime.now(), '%Y-%m-%d')
previous_30_days_train = datetime.strftime(datetime.strptime(today, '%Y-%m-%d') - timedelta(days=30), '%Y-%m-%d')
firstDayThisMonth = datetime.strftime(datetime.now(), '%Y-%m-') + '01'
# Từ điển giải thích features
with open('persistentvolume/datas/dictionary_features.json', "rb") as fp:
    dict_features = json.loads(fp.read())

# Từ điển giải thích variable
with open('persistentvolume/datas/dictionary_variables.json', "rb") as fp:
    dict_variables = json.loads(fp.read())
# =========================================================================================
lastDayPreviousMonth = datetime.strftime(datetime.strptime(firstDayThisMonth, '%Y-%m-%d')-timedelta(days = 1), '%Y-%m-%d')
previous_30_days = datetime.strftime(datetime.strptime(firstDayThisMonth, '%Y-%m-%d')-timedelta(days = 31), '%Y-%m-%d')
month = datetime.strftime(datetime.strptime(lastDayPreviousMonth, '%Y-%m-%d'), "%Y%m") # Chỉ sử dụng cho training

MODEL_FILE_LGB = 'persistentvolume/models/lgb_classifier_all.pkl'
MODEL_FILE_RDF = r'persistentvolume/models/rd_classifier_all.pkl'
MODEL_FILE_MLP = 'persistentvolume/models/mlp_classifier_all.h5'
MODEL_FILE_SVM = 'persistentvolume/models/svm_classifier_all.pkl'
MODEL_FILE_DEC = 'persistentvolume/models/dec_classifier_all.pkl'

MODEL_FILE_LGB_20 = 'persistentvolume/models/lgb_classifier_20.pkl'
MODEL_FILE_RDF_20 = r'persistentvolume/models/rd_classifier_20.pkl'
MODEL_FILE_MLP_20 = 'persistentvolume/models/mlp_classifier_20.h5'
MODEL_FILE_SVM_20 = 'persistentvolume/models/svm_classifier_20.pkl'
MODEL_FILE_DEC_20 = 'persistentvolume/models/dec_classifier_20.pkl'

PREFIX_MODEL_FILE_LGB = 'persistentvolume/models/lgb_classifier_all'
PREFIX_MODEL_FILE_RDF = r'persistentvolume/models/rd_classifier_all'
PREFIX_MODEL_FILE_MLP = 'persistentvolume/models/mlp_classifier_all'
PREFIX_MODEL_FILE_SVM = 'persistentvolume/models/svm_classifier_all'
PREFIX_MODEL_FILE_DEC = 'persistentvolume/models/dec_classifier_all'

PREFIX_MODEL_FILE_LGB_20 = 'persistentvolume/models/lgb_classifier_20'
PREFIX_MODEL_FILE_RDF_20 = r'persistentvolume/models/rd_classifier_20'
PREFIX_MODEL_FILE_MLP_20 = 'persistentvolume/models/mlp_classifier_20'
PREFIX_MODEL_FILE_SVM_20 = 'persistentvolume/models/svm_classifier_20'
PREFIX_MODEL_FILE_DEC_20 = 'persistentvolume/models/dec_classifier_20'

PREFIX_X_TRAIN = 'persistentvolume/datas/X_train'
PREFIX_Y_TRAIN = 'persistentvolume/datas/y_train'
PREFIX_X_TEST = 'persistentvolume/datas/X_test'
PREFIX_Y_TEST = 'persistentvolume/datas/y_test'

DATA_ORDER_INPUT_30 = 'persistentvolume/datas/data_input'
# List các cột của DF_SUMMARY_DUMMY
FINAL_FEATURES = 'persistentvolume/datas/final_features'
SCALER_FILE = 'persistentvolume/models/scaler'
# importance features list (chnhs là index của bảng DF_IMPORTANCE_FEATURE)
IMPORTANCE_FEATURE = 'persistentvolume/datas/importance_features'
# Bảng final features chứa features, variable của feature thuộc về, fraudGroup của nó.
DF_FINAL_FEATURE = 'persistentvolume/datas/df_final_features'
# Bảng importance_feature từ random forest
DF_IMPORTANCE_FEATURE = 'persistentvolume/datas/df_importance_features'
# Bảng features cuối cùng chứa các dummy variable (dùng để split train và test)
DF_SUMMARY_DUMMY = 'persistentvolume/datas/df_summary_dummy'
# Bảng correlation
DF_CORRELATION = 'persistentvolume/datas/df_correlation'


# List các columns nếu đó là dữ liệu training (bao gồm nhãn y đánh dấu thông qua trường FRAUD_SaveValue_Check)
COLUMNS_QUERY = ['ORDER_DATE', 'ORDER_CODE', 'PAYMENTMODES', 'PAYMENTSTATUS',
'EMAIL_ACCOUNT', 'USER_ID', 'PHONE_DELIVERY', 'EMAIL_DELIVERY',
'P_STREETNUMBER', 'STREET', 'WARD', 'VNDISTRICT', 'PROVINCE',
'OFFER_CODE', 'OFFER_NAME', 'MERCHANTCODE', 'MERCHANTNAME', 'WAREHOUSE',
'P_QUANTITY', 'AFF_COMMISSION', 'AFF_PARTNERNAME', 'AFF_UTMSOURCE',
'SUB2_MAIN', 'SUB3_MAIN', 'SUB4_MAIN', 'SUB5_MAIN', 'SUB6_MAIN',
'SUB7_MAIN', 'CODE2_MAIN', 'CODE3_MAIN', 'CODE4_MAIN', 'CODE5_MAIN',
'CODE6_MAIN', 'CODE7_MAIN', 'Item_PK', 'OFFER_FINAL_PRICE',
'OFFER_PRICE', 'P_SUBTOTAL', 'P_TOTALPRICE', 'DISCOUNT_AMOUNT',
'P_TOTALORDERAMOUNT', 'ORDER_USER_VINIDACCOUNT', 'ORDER_VINIDNUMBER',
'ORDER_CUSTOMER_TYPE', 'SALES_APPLICATION', 'IP_ADDRESS', 'FRAUD_SaveValue_Check', 'TTKD']

# List các columns nếu đó là dữ liệu đơn hàng trong ngày (không có FRAUD_SaveValue_Check)
COLUMNS_QUERY_NEW_ORDER = ['ORDER_DATE', 'ORDER_CODE', 'PAYMENTMODES', 'PAYMENTSTATUS',
'EMAIL_ACCOUNT', 'USER_ID', 'PHONE_DELIVERY', 'EMAIL_DELIVERY',
'P_STREETNUMBER', 'STREET', 'WARD', 'VNDISTRICT', 'PROVINCE',
'OFFER_CODE', 'OFFER_NAME', 'MERCHANTCODE', 'MERCHANTNAME', 'WAREHOUSE',
'P_QUANTITY', 'AFF_COMMISSION', 'AFF_PARTNERNAME', 'AFF_UTMSOURCE',
'SUB2_MAIN', 'SUB3_MAIN', 'SUB4_MAIN', 'SUB5_MAIN', 'SUB6_MAIN',
'SUB7_MAIN', 'CODE2_MAIN', 'CODE3_MAIN', 'CODE4_MAIN', 'CODE5_MAIN',
'CODE6_MAIN', 'CODE7_MAIN', 'Item_PK', 'OFFER_FINAL_PRICE',
'OFFER_PRICE', 'P_SUBTOTAL', 'P_TOTALPRICE', 'DISCOUNT_AMOUNT',
'P_TOTALORDERAMOUNT', 'ORDER_USER_VINIDACCOUNT', 'ORDER_VINIDNUMBER',
'ORDER_CUSTOMER_TYPE', 'SALES_APPLICATION', 'IP_ADDRESS', 'TTKD']


# List các columns sau khi đi qua preprocessing step 1 (thêm 'DATE', 'TIME', 'TIMEUTC', 'AFF_EST_COST')
COLUMNS_QUERY_INPUT = ['ORDER_DATE', 'ORDER_CODE', 'PAYMENTMODES', 'PAYMENTSTATUS',
       'EMAIL_ACCOUNT', 'USER_ID', 'PHONE_DELIVERY', 'EMAIL_DELIVERY',
       'P_STREETNUMBER', 'STREET', 'WARD', 'VNDISTRICT', 'PROVINCE',
       'OFFER_CODE', 'OFFER_NAME', 'MERCHANTCODE', 'MERCHANTNAME', 'WAREHOUSE',
       'P_QUANTITY', 'AFF_COMMISSION', 'AFF_PARTNERNAME', 'AFF_UTMSOURCE',
       'SUB2_MAIN', 'SUB3_MAIN', 'SUB4_MAIN', 'SUB5_MAIN', 'SUB6_MAIN',
       'SUB7_MAIN', 'CODE2_MAIN', 'CODE3_MAIN', 'CODE4_MAIN', 'CODE5_MAIN',
       'CODE6_MAIN', 'CODE7_MAIN', 'Item_PK', 'OFFER_FINAL_PRICE',
       'OFFER_PRICE', 'P_SUBTOTAL', 'P_TOTALPRICE', 'DISCOUNT_AMOUNT',
       'P_TOTALORDERAMOUNT', 'ORDER_USER_VINIDACCOUNT', 'ORDER_VINIDNUMBER',
       'ORDER_CUSTOMER_TYPE', 'SALES_APPLICATION', 'IP_ADDRESS', 'TTKD', 'DATE', 'TIME', 'TIMEUTC',
       'AFF_EST_COST']

# List các columns sau khi biến đổi các biến category sang one-hot encoding
COLUMNS_DUMMY_INPUT = ['ORDER_CODE', 'AFF_COMMISSION', 'AFF_COMMISSION_FEE', 'AFF_EST_COST',
'DISCOUNT_AMOUNT', 'OFFER_FINAL_PRICE', 'OFFER_PRICE', 'P_QUANTITY', 'P_SUBTOTAL', 'P_TOTALORDERAMOUNT',
'P_TOTALPRICE', 'IP_SUSPECT_30_2', 'IP_SUSPECT_30_5', 'IP_SUSPECT_30_10', 'IP_SUSPECT_30_20',
'IP_SUSPECT_WEEK0', 'IP_SUSPECT_WEEK1', 'IP_SUSPECT_WEEK2', 'IP_SUSPECT_WEEK3', 'IP_SUSPECT_WEEK_NUM', 'IP_SUSPECT_WEEK', 'N_CODE2_MAIN',
'N_CODE5_MAIN', 'N_OFFER_CODE', 'IP_N_CODE2_MAIN', 'IP_N_CODE5_MAIN', 'IP_N_EMAIL_ACCOUNT', 'IP_N_OFFER_CODE',
'IP_N_ORDER_VINACCOUNT', 'IP_N_PHONE', 'IP_STD_TIMEUTC', 'IP_N_WARD', 'PHONE_N_CODE2_MAIN', 'PHONE_N_CODE5_MAIN',
'PHONE_N_EMAIL_ACCOUNT', 'PHONE_N_IP_ADDRESS', 'PHONE_N_OFFER_CODE', 'PHONE_N_ORDER_VINACCOUNT', 'PHONE_STD_TIMEUTC',
'PHONE_N_WARD', 'ORDER_VINACC_N_CODE2_MAIN', 'ORDER_VINACC_N_CODE5_MAIN', 'ORDER_VINACC_N_EMAIL_ACCOUNT',
'ORDER_VINACC_N_IP_ADDRESS', 'ORDER_VINACC_N_OFFER_CODE', 'ORDER_VINACC_N_PHONE', 'ORDER_VINACC_STD_TIMEUTC',
'ORDER_VINACC_N_WARD', 'IS_VINID_MATCH', 'IS_AFF_USE', 'IS_MATCH_EMAIL', 'IS_ADR_EMAIL_ACCOUNT',
'PAYMENTMODES_GROUP_ADR_PLC', 'PAYMENTMODES_GROUP_BANK_TRANSFER', 'PAYMENTMODES_GROUP_COD', 'PAYMENTMODES_GROUP_DISCOUNT',
'PAYMENTMODES_GROUP_EWALLET', 'PAYMENTMODES_GROUP_EXCHANGE_ORDER', 'PAYMENTMODES_GROUP_GiftCode', 'PAYMENTMODES_GROUP_INVOICE',
'PAYMENTMODES_GROUP_NAPAS', 'PAYMENTMODES_GROUP_ONEPAY', 'PAYMENTMODES_GROUP_QR_APP_V1', 'PAYMENTMODES_GROUP_TRA_GOP',
'PAYMENTMODES_GROUP_VinID', 'PAYMENTSTATUS_FAILED', 'PAYMENTSTATUS_MAKING_PAYMENT', 'PAYMENTSTATUS_NOTPAID',
'PAYMENTSTATUS_PAID', 'PAYMENTSTATUS_PARTPAID', 'SALES_APPLICATION_Web', 'SALES_APPLICATION_WebMobile',
'SALES_APPLICATION_mobile_android', 'SALES_APPLICATION_mobile_ios', 'AFF_PARTNERNAME_ADR blog', 'AFF_PARTNERNAME_Adayroi',
'AFF_PARTNERNAME_Coccoc','AFF_PARTNERNAME_Criteo', 'AFF_PARTNERNAME_Direct', 'AFF_PARTNERNAME_Email', 'AFF_PARTNERNAME_Facebook',
'AFF_PARTNERNAME_Google', 'AFF_PARTNERNAME_HasOffer', 'AFF_PARTNERNAME_RTB House', 'AFF_PARTNERNAME_Zalo']

# Nhóm biến theo ORDER_KEY
ORDER_COLS_MEA = ['P_SUBTOTAL', 'P_TOTALPRICE', 'P_TOTALORDERAMOUNT', 'TIMEUTC']
ORDER_COLS_DIM_TRAIN = ['ORDER_DATE', 'ORDER_CODE', 'PAYMENTMODES',
                      'PAYMENTSTATUS', 'EMAIL_ACCOUNT', 'USER_ID', 'PHONE_DELIVERY',
                      'EMAIL_DELIVERY', 'P_STREETNUMBER', 'STREET', 'WARD', 'VNDISTRICT',
                      'PROVINCE', 'ORDER_USER_VINIDACCOUNT', 'ORDER_VINIDNUMBER',
                      'SALES_APPLICATION', 'IP_ADDRESS',
                      'FRAUD_SaveValue_Check', 'DATE', 'TIME']

ORDER_COLS_DIM_NEW = ['ORDER_DATE', 'ORDER_CODE', 'PAYMENTMODES',
                      'PAYMENTSTATUS', 'EMAIL_ACCOUNT', 'USER_ID', 'PHONE_DELIVERY',
                      'EMAIL_DELIVERY', 'P_STREETNUMBER', 'STREET', 'WARD', 'VNDISTRICT',
                      'PROVINCE', 'ORDER_USER_VINIDACCOUNT', 'ORDER_VINIDNUMBER',
                      'SALES_APPLICATION', 'IP_ADDRESS',
                      'DATE', 'TIME']

# Nhóm biến theo ITEM_KEY
ITEM_COLS_MEA = ['P_QUANTITY', 'AFF_COMMISSION', 'OFFER_FINAL_PRICE', 'OFFER_PRICE', 'DISCOUNT_AMOUNT',
                 'AFF_EST_COST']

COL_SELECTED_TRAIN = ['ORDER_CODE', 'PAYMENTMODES_GROUP', 'PAYMENTSTATUS',
                    'SALES_APPLICATION', 'AFF_PARTNERNAME',
                    'AFF_COMMISSION', 'AFF_COMMISSION_FEE', 'AFF_EST_COST', 'DISCOUNT_AMOUNT',
                    'OFFER_FINAL_PRICE', 'OFFER_PRICE', 'P_QUANTITY', 'P_SUBTOTAL',
                    'P_TOTALORDERAMOUNT', 'P_TOTALPRICE', 'IP_SUSPECT_30_2',
                    'IP_SUSPECT_30_5', 'IP_SUSPECT_30_10', 'IP_SUSPECT_30_20',
                    'IP_SUSPECT_WEEK0', 'IP_SUSPECT_WEEK1', 'IP_SUSPECT_WEEK2',
                    'IP_SUSPECT_WEEK3', 'IP_SUSPECT_WEEK_NUM', 'IP_SUSPECT_WEEK',
                    'N_CODE2_MAIN', 'N_CODE5_MAIN', 'N_OFFER_CODE', 'IP_N_CODE2_MAIN',
                    'IP_N_CODE5_MAIN', 'IP_N_EMAIL_ACCOUNT', 'IP_N_OFFER_CODE',
                    'IP_N_ORDER_VINACCOUNT', 'IP_N_PHONE', 'IP_STD_TIMEUTC', 'IP_N_WARD',
                    'PHONE_N_CODE2_MAIN', 'PHONE_N_CODE5_MAIN', 'PHONE_N_EMAIL_ACCOUNT',
                    'PHONE_N_IP_ADDRESS', 'PHONE_N_OFFER_CODE', 'PHONE_N_ORDER_VINACCOUNT',
                    'PHONE_STD_TIMEUTC', 'PHONE_N_WARD',
                    'ORDER_VINACC_N_CODE2_MAIN', 'ORDER_VINACC_N_CODE5_MAIN', 'ORDER_VINACC_N_EMAIL_ACCOUNT',
                    'ORDER_VINACC_N_IP_ADDRESS', 'ORDER_VINACC_N_OFFER_CODE',
                    'ORDER_VINACC_N_PHONE', 'ORDER_VINACC_STD_TIMEUTC',
                    'ORDER_VINACC_N_WARD', 'IS_VINID_MATCH',
                    'IS_AFF_USE', 'IS_MATCH_EMAIL',
                    'IS_ADR_EMAIL_ACCOUNT', 'FRAUD_SaveValue_Check']

COL_SELECTED_NEW_ORDER = ['ORDER_CODE', 'PAYMENTMODES_GROUP', 'PAYMENTSTATUS',
                    'SALES_APPLICATION', 'AFF_PARTNERNAME',
                    'AFF_COMMISSION', 'AFF_COMMISSION_FEE', 'AFF_EST_COST', 'DISCOUNT_AMOUNT',
                    'OFFER_FINAL_PRICE', 'OFFER_PRICE', 'P_QUANTITY', 'P_SUBTOTAL',
                    'P_TOTALORDERAMOUNT', 'P_TOTALPRICE', 'IP_SUSPECT_30_2',
                    'IP_SUSPECT_30_5', 'IP_SUSPECT_30_10', 'IP_SUSPECT_30_20',
                    'IP_SUSPECT_WEEK0', 'IP_SUSPECT_WEEK1', 'IP_SUSPECT_WEEK2',
                    'IP_SUSPECT_WEEK3', 'IP_SUSPECT_WEEK_NUM', 'IP_SUSPECT_WEEK',
                    'N_CODE2_MAIN', 'N_CODE5_MAIN', 'N_OFFER_CODE', 'IP_N_CODE2_MAIN',
                    'IP_N_CODE5_MAIN', 'IP_N_EMAIL_ACCOUNT', 'IP_N_OFFER_CODE',
                    'IP_N_ORDER_VINACCOUNT', 'IP_N_PHONE', 'IP_STD_TIMEUTC', 'IP_N_WARD',
                    'PHONE_N_CODE2_MAIN', 'PHONE_N_CODE5_MAIN', 'PHONE_N_EMAIL_ACCOUNT',
                    'PHONE_N_IP_ADDRESS', 'PHONE_N_OFFER_CODE', 'PHONE_N_ORDER_VINACCOUNT',
                    'PHONE_STD_TIMEUTC', 'PHONE_N_WARD',
                    'ORDER_VINACC_N_CODE2_MAIN', 'ORDER_VINACC_N_CODE5_MAIN', 'ORDER_VINACC_N_EMAIL_ACCOUNT',
                    'ORDER_VINACC_N_IP_ADDRESS', 'ORDER_VINACC_N_OFFER_CODE',
                    'ORDER_VINACC_N_PHONE', 'ORDER_VINACC_STD_TIMEUTC',
                    'ORDER_VINACC_N_WARD', 'IS_VINID_MATCH',
                    'IS_AFF_USE', 'IS_MATCH_EMAIL',
                    'IS_ADR_EMAIL_ACCOUNT']


RETURN_HYBRIS = {
                    'PHONE_N_ORDER_VINACCOUNT':'SL thẻ VINID',
                    'PHONE_N_EMAIL_ACCOUNT':'SL email',
                    'PHONE_N_IP_ADDRESS':'SL IP',
                    'PHONE_N_OFFER_CODE':'SL đơn hàng',
                    'PHONE_STD_TIMEUTC':'Khoảng cách thời gian đặt hàng (ngày)',
                    'PHONE_N_WARD':'SL Phường/Xã',
                    'PHONE_N_CODE2_MAIN':'SL sản phẩm maincate cấp 2',
                    'ORDER_VINACC_N_OFFER_CODE':'SL đơn hàng',
                    'ORDER_VINACC_N_PHONE':'SL SĐT người nhận hàng',
                    'ORDER_VINACC_N_CODE2_MAIN':'SL maincate cấp 2',
                    'ORDER_VINACC_N_IP_ADDRESS':'SL địa chỉ IP',
                    'AFF_EST_COST':'Chi phí hoa hồng dự kiến',
                    'AFF_COMMISSION_FEE':'Mức phí hoa hồng trung bình',
                    'AFF_PARTNERNAME':'Affiliate partner',
                    'IP_N_EMAIL_ACCOUNT':'SL Email nhận hàng',
                    'IP_N_PHONE':'SL SĐT nhận hàng',
                    'IP_N_OFFER_CODE':'SL đơn hàng',
                    'IP_SUSPECT_WEEK_NUM':'SL tuần có nhiều hơn 5 đơn hàng (30 ngày)',
                    'IP_N_CODE2_MAIN':'SL maincate cấp 2',
                    'IP_N_ORDER_VINACCOUNT':'SL thẻ VINID',
                    'IP_N_WARD':'SL Phường/Xã'
                 }

RETURN_MAP_VARGROUP = {
    'PHONE':'DeliveryPhone',
    'VIN': 'VINID',
    'COST':'Affiliate',
    'IP':'IP',
    'ORDER':'ORDER',
    'BROWSER':'BROWSER'
}
