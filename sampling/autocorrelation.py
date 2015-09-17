def AutoCorrelationFunction(samples, maxlag):
    """
    Compute autocorrelation from np.array of samples
    Ref: https://en.wikipedia.org/wiki/Autocorrelation#Estimation
    Inputs:
        samples = np.array where each row is a samples (and col = index)
        maxlag = integer giving maximum lag
    """
    ACF = []
    MEAN = samples.mean(axis=0)
    VAR = samples.var(axis=0)
    n = len(samples)
    for ii in range(1, maxlag):
        Sm = samples - MEAN
        ACF.append( (Sm[:-ii]*Sm[ii:]).sum(axis=0) / ((n-ii)*VAR) )
    return ACF
