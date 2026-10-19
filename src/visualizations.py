import matplotlib.pyplot as plt 
import seaborn as sns
import numpy as np
from sklearn.metrics import confusion_matrix, RocCurveDisplay


    
def plot_correlation_matrix(data, sufix=None, title="Macierz korelacji"):
    if sufix is not None:
        data = data.filter(regex=f"^Class$|.*{sufix}$")
    else:
        data = data.drop(columns = ['ID'])
    correlations = data.corr()
    plt.figure(figsize=(12, 10))
    sns.heatmap(correlations, annot=True, cmap='coolwarm', fmt=".2f", vmin=-1, vmax=1)
    plt.title(title)
    plt.show()
    
def plot_band_power(data, channel='Fz', hue='Class', title="Profil częstotliwościowy sygnału",
                            bands = ['Delta', 'Theta', 'Alpha', 'Beta'], ylabel="Bezwzględna moc sygnału",
                            xlabel="Pasmo częstotliwości", x='Pasmo', y='Moc'):
    band_cols = [f"{channel}_{band}" for band in bands]
    
    subset = data[band_cols + [hue]].copy()
    melted_data = subset.melt(id_vars=hue, var_name=x, value_name=y)
    
    plt.figure(figsize=(8, 6))
    sns.barplot(data=melted_data, x=x, y=y, hue=hue, errorbar='sd', alpha=0.9)
    
    plt.title(title)
    plt.ylabel(ylabel)
    plt.xlabel(xlabel)
    
    plt.show()
    
def plot_box(data, x='Class', y='Fz_TBR', title="Rozkład wskaźnika Theta/Beta Ratio (TBR) w kanale Fz", 
             xlabel= "0 = grupa kontrolna, 1 = ADHD", ylabel = "wskaźnik TBR"):
    plt.figure(figsize=(6, 5))
    sns.boxplot(data=data, x=x, y=y)
    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.show()

def plot_confusion_matrix(y_test, y_pred, title = "Macierz błędów"):
    cm = confusion_matrix(y_test, y_pred)
    plt.figure(figsize=(5, 4))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                xticklabels=['Control', 'ADHD'], 
                yticklabels=['Control', 'ADHD'])
    plt.ylabel('Prawdziwa klasa')
    plt.xlabel('Przewidziana klasa')
    plt.title(title)
    plt.show()  
    
def plot_roc(model, X_test, y_test, title="Krzywa ROC"):
    
    fig, ax = plt.subplots(figsize=(6, 5))
    RocCurveDisplay.from_estimator(model, X_test, y_test, ax=ax)

    ax.plot([0, 1], [0, 1], color='pink', linestyle='--', label='Losowy wybór')
    
    plt.title(title)
    plt.legend()
    plt.show()
    
def plot_importance(pipeline, feature_names, top_n=10, title="Top 10 najważniejszych cech"):
   
    model = pipeline.named_steps['classifier']
   
    importances = model.feature_importances_
    
    indices = np.argsort(importances)[::-1][:top_n]
    top_features = [feature_names[i] for i in indices]
    top_importances = importances[indices]
    
    plt.figure(figsize=(8, 6))
    sns.barplot(x=top_importances, y=top_features)
    plt.title(title)
    plt.xlabel('Względna ważność cechy')
    plt.ylabel('Cecha')
    plt.show()