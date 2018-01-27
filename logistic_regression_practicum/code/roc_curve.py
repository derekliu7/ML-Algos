def roc_curve(probabilities, labels):
    '''
    INPUT: numpy array, numpy array
    OUTPUT: list, list, list

    Take a numpy array of the predicted probabilities and a numpy array of the
    true labels.
    Return the True Positive Rates, False Positive Rates and Thresholds for the
    ROC curve.
    '''
    list_prob = probabilities.tolist()
    list_lab = labels.tolist()
    thresh = sorted(list_prob)
    tub = zip(list_prob, list_lab)
    tprs = []
    fprs = []
    for x in thresh:
        tp = sum([1 for (p, l) in tub if l > 0 and p > x])
        fp = sum([1 for (p, l) in tub if l <= 0 and p > x])
        tprs.append(tp)
        fprs.append(fp)

    return tprs, fprs, thresh
