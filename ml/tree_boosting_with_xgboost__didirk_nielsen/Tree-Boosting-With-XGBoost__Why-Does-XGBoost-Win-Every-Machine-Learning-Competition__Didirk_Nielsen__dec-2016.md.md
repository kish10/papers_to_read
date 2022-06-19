# Tree Boosting With XGBoost - Why Does XGBoost Win "Every" Machine Learning Competition?
Didrik Nielsen
(http://pzs.dstu.dp.ua/DataMining/boosting/bibl/Didrik.pdf)

## 9 Why Does XGBoost Win "Every" Competition?

### 9.1 Boosting With Tree Stumps in One Dimension

'additive tree models have rich representational abilities.'
- 'Although they are constrained to be piecewise constant functions, they can potentially approximate any function arbitrarily close given that enough trees are included.'
- 'The perhaps greatest benefit however is that tree boosting can be seen to adaptively determine the local neighbourhoods.'

#### 9.1.1 Adaptive neighbourhoods

'Let us consider a regression task and compare additive tree models with local linear regression and smoothing splines.'
- 'All of these methods have regularization parameters which allows one to adjust the flexibility of the fit to the data'
  - 'However,' 
    - 'while methods such as local regression and smoothing splines can be seen to use the same amount of flexibility in the whole input space of X,
    - 'additive tree models can be seen to adjust the amount of flexibility locally in X to the amount which seems necessary.' 
      - 'That is, additive tree models can be seen to adaptively determine the size of the local neighbourhoods.'

A sample of 500 was generated from:
- ![9_1_1__function_to_generate_samples.jpg](/ml/tree_boosting_with_xgboost__didirk_nielsen/images/9_1_1__function_to_generate_samples.jpg)
- Sample:
  - ![9_1_1__figure_9_1__generated_sample.jpg](/ml/tree_boosting_with_xgboost__didirk_nielsen/images/9_1_1__figure_9_1__generated_sample.jpg)

For both local linear regression models & smoothing splines:
- Less flexible versions of the models are unable to capture the more complex structure in higher values of x, but do a job of capturing the structure in the lower values of x
  - Note: Less flexible versions use larger neighborhood sizes
- More flexible versions of the models can capture the structure in the more higher, but 'seems to flexible for lower values of x'
  - Note: More flexible versions use smaller neighborhood sizes
- Fit of local linear regression models:
  - ![9_1_1__figure_9_2__local_linear_regression_fit_of_sample.jpg](/ml/tree_boosting_with_xgboost__didirk_nielsen/images/9_1_1__figure_9_2__local_linear_regression_fit_of_sample.jpg)
- Fit of smoothing spline models:
  - ![9_1_1__figure_9_3__smoothing_splines_fit_of_sample.jpg](/ml/tree_boosting_with_xgboost__didirk_nielsen/images/9_1_1__figure_9_3__smoothing_splines_fit_of_sample.jpg)

Whereas boosted tree models are able to adequately capture the structure in the whole range of x, from which we can see the boosted trees to make use of 'adaptive neighbourhood sizes on what seems necessary from the data'.
- 'In areas where complex structure is apparent from the data, smaller neighbourhoods are used, whereas in areas where complex structure seems to be lacking, a wider neighbourhood is used'

### 9.1.2 The Weight Function Interpretation

In order 'to better understand the nature of how local neighbourhoods are determined', by making 'the notion of the neighbourhood more concrete'.
- 'We will do this by considering the interpretation that many models, including additive tree models, local regression and smoothing splines, can be seen to make predictions using a weighted average of the training data.'



