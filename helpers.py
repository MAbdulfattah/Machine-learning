import pandas as pd
import numpy as np
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn import metrics
from tensorflow.keras import models, layers, optimizers
import matplotlib.pyplot as plt 


def plot_roc(model, x, y, name='linear'):
    y_text = model.predict_proba(x)

    curve_linear = metrics.roc_curve(y, y_text[:, 1])

    # Compute AUC from the created ROC
    auc_linear = metrics.auc(curve_linear[0], curve_linear[1])

    plt.plot(curve_linear[0], curve_linear[1], label=f'{name} (area = %0.2f)' % auc_linear)

    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.0])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('ROC curve');

    plt.legend();


def calculate_accuracy(XTest, yTest, model):
    X_test_final, _, y_test_final, _ = \
            train_test_split(XTest, yTest, test_size=0.5, shuffle=True)
    predict_test_final = model.predict_classes(X_test_final)
    accuracy = metrics.accuracy_score(np.argmax(y_test_final, 1), predict_test_final)
    return accuracy

def make_norm(X_test, y_test, clf):
    accuracies = []    
    for i in range(1000):
        accuracy = calculate_accuracy(X_test, y_test, clf)
        accuracies.append(round(accuracy, 3))
    return accuracies
    
def plot_results(accuracy, label='Accuracy'):
    df_accuracy = pd.DataFrame(accuracy, columns =[label])
    sns_plot = sns.histplot(data=df_accuracy, x=label,kde=True)
    
    
def create_lstm_model(wordIndex):
    model= models.Sequential()
    model.add(layers.Embedding(wordIndex, 100, input_length=30))
    model.add(layers.LSTM(100, dropout=0.2, recurrent_dropout=0.2))
    model.add(layers.Dense(2, activation='sigmoid'))

    model.compile(loss='binary_crossentropy', optimizer=optimizers.Adam(lr=0.001),
                  metrics=['accuracy'])
    return model

def old_shit():
    his_df = pd.read_csv('val_accuracy_lstm.csv')
    # TODO: choose the method Sybolt made. 
    his_df_file = [{"history": { "val_accuracy": h }} for h in his_df.values]

    his_df_file = pd.DataFrame(his_df.iloc[:,-1].values, columns =['Accuracy'])
    sns_plot = sns.histplot(data=his_df_file, x="Accuracy",kde=True)
    sns_plot.set(xlabel='Validation Accuracy')
    sns_plot.figure.savefig("lstm_val_acc.png")
    ss.shapiro(his_df_file['Accuracy'])