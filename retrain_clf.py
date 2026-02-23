"""
retrain cytotoxicity classifier on custom dataset
"""

import argparse
from pathlib import Path
import pandas as pd
import numpy as np
import pickle
import joblib
import seaborn as sns
import matplotlib.pyplot as plt

from sklearn.ensemble import RandomForestClassifier as rfc
from sklearn.metrics import accuracy_score, matthews_corrcoef, f1_score, precision_score, recall_score, confusion_matrix

from src.config import OUTPUT_PATH, DEVICE
from src.utils.gen_esm_embedds import Embedder

def validate(clf:rfc, train_data, val_data, test_data, outdir:Path):

    def calculate_metrics(y_true, y_score, thr:float = 0.5, title:str='retraining_evaluation'):

        y_pred = y_score >= thr
        accuracy = accuracy_score(y_true, y_pred)
        f1 = f1_score(y_true, y_pred)
        mcc = matthews_corrcoef(y_true, y_pred)
        precision = precision_score(y_true, y_pred)
        recall = recall_score(y_true, y_pred)

        result_metrics = {
            'accuracy': accuracy,
            'f1': f1,
            'mcc': mcc,
            'precision': precision,
            'recall': recall
        }

        c_matrix = confusion_matrix(y_true, y_pred)
        result_metrics['confusion_matrix'] = c_matrix

        # Set up a figure with two subplots
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5), gridspec_kw={'width_ratios': [1, 1.5]})
        fig.suptitle(title,fontsize=16)
        # Bar plot for metrics
        metrics_names = ['Accuracy', 'Precision', 'Recall', 'F1-Score']
        metrics_values = [result_metrics['accuracy'], result_metrics['precision'], 
                          result_metrics['recall'], result_metrics['f1']]
        bars = ax1.bar(metrics_names, metrics_values, color=['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728'])
        ax1.set_ylim(0, 1)
        ax1.set_title('Classification Metrics', fontsize=14)
        ax1.set_ylabel('Score', fontsize=14)
        ax1.tick_params(axis='x', rotation=45)

        # Add value labels on top of bars
        for bar in bars:
            yval = bar.get_height()
            ax1.text(bar.get_x() + bar.get_width()/2, yval + 0.02, f'{yval:.3f}', 
                     ha='center', va='bottom', fontsize=12)

        # Heatmap for confusion matrix
        sns.heatmap(c_matrix, annot=True, fmt='d', cmap='Blues', ax=ax2, 
                    cbar_kws={'label': 'Count'}, annot_kws={'size': 14})
        ax2.set_title('Confusion Matrix', fontsize=16)
        ax2.set_xlabel('Predicted Label', fontsize=14)
        ax2.set_ylabel('True Label', fontsize=14)

        # Adjust layout and display
        plt.tight_layout()
        plt.savefig(outdir / f"{title}.png")

    """
    - Compute performance on train, validation and test sets
    - visualize results 
    """

    X_train, y_train = train_data['arr_0'], train_data['arr_1']
    X_val, y_val = val_data['arr_0'], val_data['arr_1']
    X_test, y_test = test_data['arr_0'], test_data['arr_1']

    prob_train = clf.predict_proba(X_train)[:,1]
    prob_val = clf.predict_proba(X_val)[:,1]
    prob_test = clf.predict_proba(X_test)[:,1]

    calculate_metrics(y_train,prob_train,title='evaluation_train')
    calculate_metrics(y_val,prob_val,title='evaluation_val')
    #calculate_metrics(y_train,prob_train,'evaluation_train')

def main(args):

    training_folder = args.training_folder
    output_folder = args.output_folder
    device = args.device

    ### CREATE OUTPUT DIRECTORY ###

    output_folder.mkdir(parents=True, exist_ok=True)

    ### DATA PREPROCESSING ###

    # load train, test, val datasets
    print('loading training data')
    train = pd.read_csv(training_folder / 'train_set.csv',index_col=False)
    test = pd.read_csv(training_folder / 'test_set.csv',index_col=False)
    val = pd.read_csv(training_folder / 'validation_set.csv',index_col=False)

    # generate embeddings
    print('\n generating embeddings')
    embedder = Embedder(device)
    train_embeddings = embedder.get_embeddings(train['sequence'].to_list())
    test_embeddings = embedder.get_embeddings(test['sequence'].to_list())
    val_embeddings = embedder.get_embeddings(val['sequence'].to_list())

    # get labels
    train_labels = train['label'].to_numpy()
    test_labels = test['label'].to_numpy()
    val_labels = val['label'].to_numpy()

    # save embeddings and labels
    np.savez(output_folder / 'train.npz',train_embeddings,train_labels)
    np.savez(output_folder / 'test.npz',test_embeddings,test_labels)
    np.savez(output_folder / 'val.npz',val_embeddings,val_labels)

    ### MODEL TRAINING ###

    train_data = np.load(output_folder / 'train.npz')
    test_data = np.load(output_folder / 'test.npz')
    val_data = np.load(output_folder / 'val.npz')

    X_train, y_train = train_data['arr_0'], train_data['arr_1']
    X_test, y_test = test_data['arr_0'], test_data['arr_1']
    X_val, y_val = val_data['arr_0'], val_data['arr_1']

    print(f'training data: {len(X_train)}')
    print(f'testin data: {len(X_test)}')
    print(f'validation data: {len(X_val)}')

    new_forest_clf = rfc(
        n_estimators=50,
        max_depth=15,
        criterion='gini',
        min_samples_leaf=1
    ).fit(X_train,y_train)

    ### MODEL EVALUATION + SAVING ###
    print(f'evaluating and saving results to {str(output_folder)}')
    validate(
        clf=new_forest_clf,
        train_data=train_data,
        test_data=test_data,
        val_data=val_data,
        outdir=output_folder
    )

    joblib.dump(new_forest_clf,output_folder / 'rtr_ctt.pkl')

def parse_args():

    parser = argparse.ArgumentParser()

    parser.add_argument(
        '--training_folder',
        type=Path,
        help="path to folder with train, validation and test set"
    )

    parser.add_argument(
        '--output_folder',
        type=Path,
        default= OUTPUT_PATH / 'CTT_RETRAINING',
        help="where to save newly trained model (.pkl format) and all metadata"
    )

    parser.add_argument(
        '--device',
        type=str,
        choices=['cpu', 'cuda', 'mps'],
        default=DEVICE,
        help="device for computation"
    )

    args = parser.parse_args()
    return args

if __name__ == '__main__':
    
    args = parse_args()
    main(args)