#!/usr/bin/python


def outlierCleaner(predictions, ages, net_worths):
    """
        Clean away the 10% of points that have the largest
        residual errors (difference between the prediction
        and the actual net worth).

        Return a list of tuples named cleaned_data where 
        each tuple is of the form (age, net_worth, error).
    """
    
    cleaned_data = []

    ### your code goes here
    error = []
    i = 0
    for actual, pred in zip(net_worths, predictions):
        diff = actual[0] - pred[0];
        error.append((i, abs(diff)))
        i += 1

    errors_sorted = sorted(error, key=lambda err: err[1])
    for i in range(0, int(len(errors_sorted) * 0.9)):
        idx = errors_sorted[i][0]
        cleaned_data.append((ages[idx], net_worths[idx], error[idx]))
    
    return cleaned_data

