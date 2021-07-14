# feature-selection
Three feature selection methods including variance threshold, correlation threshold and forward search. Tested on data from sci-kit learn library.

1. Variance threshold:
  Removes features whose variance falls below a threshold.
2. Correlation threshold:
  Removes features whose correlation falls below a threshold. Uses Pearson's correlation.
3. Forward search
  Wrapper-based method, basically brute-force. In each step it finds the best performing feature for the current set.

Datasets:
  Iris (4 features)
  Wine (13 features)
  Breast cancer (30 features)
