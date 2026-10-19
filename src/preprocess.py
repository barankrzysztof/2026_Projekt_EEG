import pandas as pd
from sklearn.model_selection import StratifiedGroupKFold

def get_subject_split(data, target_col='Class', group_col='ID', n_splits=5, random_state=67):

    # stratified group k fold łączy w sobie group fold, czyli ten sam pacjent nie trafi
    # do zbioru treningowego i testowego oraz stratyfikację, czyli proporcja klas w zbiorze
    # testowym i treningowym zostanie zachowana. dzielimy na 5 koszyków, czyli w proporcji 80/20
    sgkf = StratifiedGroupKFold(n_splits=n_splits, shuffle=True, random_state=random_state)
    
    # bierzemy tylko pierwszy zbiór
    train_idx, test_idx = next(sgkf.split(data, y=data[target_col], groups=data[group_col]))
    
    train_data = data.iloc[train_idx].copy()
    test_data = data.iloc[test_idx].copy()
    
    X_train = train_data.drop(columns=[target_col])
    y_train = train_data[target_col]
    
    X_test = test_data.drop(columns=[target_col])
    y_test = test_data[target_col]
    
    return X_train, X_test, y_train, y_test

def map_class(data):
    data = data.copy()
    data['Class'] = data['Class'].map({'Control': 0, 'ADHD': 1})
    return data