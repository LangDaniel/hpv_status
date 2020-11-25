import tensorflow.keras.backend as K
import tensorflow.keras as keras

def mean_tp(y_true, y_pred):
    return K.sum(y_true*y_pred) / (K.sum(y_true) + K.epsilon())

def var_tp(y_true, y_pred):
    mean = mean_tp(y_true, y_pred)
    return K.abs(y_true * y_pred - mean) / (K.sum(y_true) + K.epsilon())

def mean_tn(y_true, y_pred):
    return K.sum((1-y_true)*y_pred) / (K.sum((1-y_true)) + K.epsilon())

def var_tn(y_true, y_pred):
    mean = mean_tn(y_true, y_pred)
    return K.abs((1-y_true) * y_pred - mean) / (K.sum(1-y_true) + K.epsilon())

def mean_differ(y_true, y_pred):
    return mean_tp(y_true, y_pred) - mean_tn(y_true, y_pred)


metrics = [ 
    keras.losses.BinaryCrossentropy(),
    keras.metrics.Precision(name='precision'),
    keras.metrics.Recall(name='recall'),
    keras.metrics.AUC(name='roc_auc'),
    keras.metrics.AUC(name='pr_auc', curve='PR'),
    mean_tp,
    mean_tn,
    mean_differ,
#    var_tp,
#    var_tn,
]
