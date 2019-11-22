from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.model_selection import train_test_split
import numpy as np

def _feature_engineering(X, transform_type = 'standard'):
    '''
    X: dataset đầu vào của model
    transform_type: dạng biến đổi của mô hình
    '''
    if transform_type == 'minmax':
        scaler = MinMaxScaler(feature_range = (-1, 1))
    elif transform_type == 'standard':
        scaler = StandardScaler()
    scaler.fit(X)
    X_scaler = scaler.transform(X)
    return scaler, X_scaler

def _train_test_split(X_input, Target, test_size = 0.2):
    '''
    X_input: Chính là X_scaler của bước trước
    Target: label của nhãn
    test_size: Tỷ lệ split data ứng với tập test trên toàn bộ model
    stratify: Biến mà theo đó các class của nó được chia đều theo tỷ lệ của test_size
    '''
    X_train, X_test, y_train, y_test, id_train, id_test = train_test_split(X_input, np.array(Target),
                                                                           np.arange(X_input.shape[0]),
                                                                           test_size = test_size,
                                                                           stratify = Target,
                                                                           random_state = 123)
    return X_train, X_test, y_train, y_test, id_train, id_test
