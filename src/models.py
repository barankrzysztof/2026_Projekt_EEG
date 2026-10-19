from sklearn.pipeline import Pipeline
from sklearn.preprocessing import RobustScaler
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.model_selection import GridSearchCV
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier


def get_pipeline(model_type='logreg'):
   
    scaler = RobustScaler() 
    
    if model_type == 'logreg':
        # uzywamy solvera liblinear, bo lepiej radzi sobie dla mocno skorelowanych danych
        # (dokladnie jak nasze sygnały) i działa szybciej
        model = LogisticRegression(class_weight='balanced', max_iter=1000, random_state=67, solver='liblinear')
    if model_type == 'svm':
            model = SVC(class_weight='balanced', max_iter=-1, random_state=67)
    if model_type == 'rf':
        model = RandomForestClassifier(class_weight='balanced', random_state=67)
    if model_type == 'ada':
            model = AdaBoostClassifier(random_state=67)
        
    return Pipeline([
        ('scaler', scaler),
        ('classifier', model)
    ])
    
    


def run_grid_search(pipeline, param_grid, X_train, y_train, cv=5, scoring='roc_auc'):
    grid_search = GridSearchCV(
        estimator=pipeline,
        param_grid=param_grid,
        cv=cv,
        scoring=scoring,
        n_jobs=-1
    )
    
    grid_search.fit(X_train, y_train)
    print(f"Najlepsze hiperparametry: {grid_search.best_params_}")
    print(f"Najlepszy wynik walidacji ({scoring}): {grid_search.best_score_:.4f}")
    
    return grid_search

