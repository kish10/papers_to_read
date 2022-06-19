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

