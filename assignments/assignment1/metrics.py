import numpy as np

def binary_classification_metrics(prediction, ground_truth):
    '''
    Computes metrics for binary classification

    Arguments:
    prediction, np array of bool (num_samples) - model predictions
    ground_truth, np array of bool (num_samples) - true labels

    Returns:
    precision, recall, f1, accuracy - classification metrics
    '''
    true_positives = ((prediction == True) & (ground_truth == True)).sum()
    false_negatives = ((prediction == False) & (ground_truth == True)).sum()
    false_positives = ((prediction == True) & (ground_truth == False)).sum()
    true_negatives = ((prediction == False) & (ground_truth == False)).sum()
    
    precision = true_positives / (true_positives + false_negatives)
    recall = true_positives / (true_positives + false_positives)
    accuracy = (true_positives + true_negatives) / ground_truth.size
    f1 = 2 * (precision * recall) / (precision + recall)

    # TODO: implement metrics!
    # Some helpful links:
    # https://en.wikipedia.org/wiki/Precision_and_recall
    # https://en.wikipedia.org/wiki/F1_score
    
    return precision, recall, f1, accuracy

def multiclass_accuracy(prediction, ground_truth):
    '''
    Computes metrics for multiclass classification

    Arguments:
    prediction, np array of int (num_samples) - model predictions
    ground_truth, np array of int (num_samples) - true labels

    Returns:
    accuracy - ratio of accurate predictions to total samples
    '''
    # TODO: Implement computing accuracy
    true_positives = 0
    true_negatives = 0
    for idx in range(ground_truth.size): 
        if (prediction[idx] == ground_truth[idx]):
            true_positives += 1
        else:
            true_negatives += 1
    
    accuracy =  true_positives / (true_positives + true_negatives)
    return accuracy
