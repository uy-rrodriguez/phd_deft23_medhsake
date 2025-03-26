Regression output comparing Humans, logits and inference as of 2025-03-27 10:49.

Regression feature coefficients sources:
 - humans: 'output/regression/linear/20250320_train+test/humans/full_no_best/regression_coefs.json'
 - logits: 'output/regression/linear/20250320_train+test/logits/full_no_best/regression_coefs.json'
 - inference: 'output/regression/linear/20250325_test/inference/full_no_best/regression_coefs.json'

Note: inference can only ever be calculated from "test" corpus.

Plots with selected features includes those which have coefficients at least
equal to the absolute mean. Regressions were done excluding years and answer
letters to exclude biases.
