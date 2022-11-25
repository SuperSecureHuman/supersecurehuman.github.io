---
title: "LazyPredict"
tags: 
    - machine learning
    - sklearn
excerpt: "I am just too lazy to compare multiple Machine Learning algorithms."
---
## LazyPredict

LazyPredict is a wrapper upon sklearn, that takes in your training data and tests it against multiple algorithms. It then returns a dataframe with the accuracy scores of each algorithm. This is useful when you are just too lazy to compare multiple algorithms.

## Installation

```bash
pip install lazypredict
```

## Usage

LazyPredict has 2 modes currently - Regressors and Classifiers

### Regressors

Regression is a supervised learning task where the goal is to predict the output of a continuous value, like a price or a probability. For example, predicting the price of a house based on the number of rooms it has.

Lets see how LazyPredict works with a regression problem.

We will be using a toy dataset from sklearn datasets

```python
from lazypredict.Supervised import LazyClassifier
from sklearn import datasets
from sklearn.model_selection import train_test_split


# Load Digits Data
digits = datasets.load_digits()

# Do test train split
X_train, X_test, y_train, y_test = train_test_split(digits.data, digits.target, test_size=0.2, random_state=42)

# Initialize LazyClassifier
clf = LazyClassifier(verbose=0,ignore_warnings=True, custom_metric=None)

# Train and Test across multiple models
models, predictions = clf.fit(X_train, X_test, y_train, y_test)
```

Now lets look at the output

```python

print(models)

```

|Model                           |Accuracy | Balanced Accuracy | ROC AUC  | F1 Score | Time Taken  |
|--------------------------------|---------|-------------------|----------|----------|-------------|
| ExtraTreesClassifier           |    0.98 |              0.98 |    None  |     0.98 |       0.14  |
| SVC                            |    0.98 |              0.98 |    None  |     0.98 |       0.04  |
| LGBMClassifier                 |    0.98 |              0.98 |    None  |     0.98 |       0.86  |
| KNeighborsClassifier           |    0.97 |              0.98 |    None  |     0.97 |       0.01  |
| RandomForestClassifier         |    0.97 |              0.97 |    None  |     0.97 |       0.12  |
| LogisticRegression             |    0.97 |              0.97 |    None  |     0.97 |       0.03  |
| XGBClassifier                  |    0.97 |              0.97 |    None  |     0.97 |       0.47  |
| CalibratedClassifierCV         |    0.97 |              0.97 |    None  |     0.97 |       0.61  |
| NuSVC                          |    0.96 |              0.96 |    None  |     0.96 |       0.04  |
| LabelPropagation               |    0.95 |              0.96 |    None  |     0.95 |       0.09  |
| LabelSpreading                 |    0.95 |              0.96 |    None  |     0.95 |       0.19  |
| PassiveAggressiveClassifier    |    0.95 |              0.95 |    None  |     0.95 |       0.04  |
| Perceptron                     |    0.96 |              0.95 |    None  |     0.95 |       0.03  |
| SGDClassifier                  |    0.95 |              0.95 |    None  |     0.95 |       0.12  |
| LinearSVC                      |    0.95 |              0.95 |    None  |     0.95 |       0.19  |
| RidgeClassifier                |    0.94 |              0.94 |    None  |     0.94 |       0.01  |
| RidgeClassifierCV              |    0.94 |              0.94 |    None  |     0.94 |       0.06  |
| LinearDiscriminantAnalysis     |    0.94 |              0.94 |    None  |     0.94 |       0.11  |
| BaggingClassifier              |    0.94 |              0.94 |    None  |     0.94 |       0.10  |
| NearestCentroid                |    0.89 |              0.88 |    None  |     0.89 |       0.01  |
| BernoulliNB                    |    0.89 |              0.88 |    None  |     0.89 |       0.02  |
| DecisionTreeClassifier         |    0.84 |              0.84 |    None  |     0.84 |       0.02  |
| QuadraticDiscriminantAnalysis  |    0.76 |              0.79 |    None  |     0.73 |       0.03  |
| ExtraTreeClassifier            |    0.77 |              0.77 |    None  |     0.77 |       0.01  |
| GaussianNB                     |    0.77 |              0.77 |    None  |     0.76 |       0.01  |
| AdaBoostClassifier             |    0.22 |              0.25 |    None  |     0.17 |       0.16  |
| DummyClassifier                |    0.08 |              0.10 |    None  |     0.01 |       0.01  |


Now with this information, we can work on improving the accuracy of our model, by taking few of the top performing models and tuning them.

### Classifiers

Classification is a supervised learning task where the goal is to predict the output of a discrete value, like a category or a label. For example, predicting whether an email is spam or not.

Classifiers are similar to regressors, there are no additional tweaks needed.

```python
from lazypredict.Supervised import LazyRegressor
from sklearn import datasets
from sklearn.model_selection import train_test_split

diabetes = datasets.load_diabetes()

# Test Train Split
X_train, X_test, y_train, y_test = train_test_split(diabetes.data, diabetes.target, test_size=0.2, random_state=42)

reg = LazyRegressor(verbose=0,ignore_warnings=True, custom_metric=None)

models, predictions = reg.fit(X_train, X_test, y_train, y_test)
```

Now looking at the output

```python
print(models)

```

|Model                          |Adjusted R-Squared | R-Squared |  RMSE | Time Taken  |
|-------------------------------|-------------------|-----------|-------|-------------|
|LarsCV                         |              0.40 |      0.47 | 52.88 |       0.02  |
|LassoLarsCV                    |              0.40 |      0.47 | 52.91 |       0.01  |
|LassoCV                        |              0.40 |      0.47 | 52.92 |       0.08  |
|OrthogonalMatchingPursuitCV    |              0.40 |      0.47 | 53.02 |       0.01  |
|ExtraTreesRegressor            |              0.40 |      0.47 | 53.07 |       0.09  |
|Lasso                          |              0.40 |      0.47 | 53.15 |       0.02  |
|PoissonRegressor               |              0.39 |      0.46 | 53.43 |       0.07  |
|ElasticNetCV                   |              0.39 |      0.46 | 53.44 |       0.05  |
|BayesianRidge                  |              0.39 |      0.46 | 53.59 |       0.01  |
|LassoLarsIC                    |              0.39 |      0.46 | 53.70 |       0.01  |
|SGDRegressor                   |              0.39 |      0.46 | 53.70 |       0.01  |
|ElasticNet                     |              0.38 |      0.45 | 53.75 |       0.01  |
|Ridge                          |              0.38 |      0.45 | 53.78 |       0.01  |
|RidgeCV                        |              0.38 |      0.45 | 53.78 |       0.01  |
|GradientBoostingRegressor      |              0.38 |      0.45 | 53.83 |       0.14  |
|TransformedTargetRegressor     |              0.38 |      0.45 | 53.85 |       0.01  |
|LinearRegression               |              0.38 |      0.45 | 53.85 |       0.01  |
|HuberRegressor                 |              0.38 |      0.45 | 54.14 |       0.01  |
|AdaBoostRegressor              |              0.37 |      0.44 | 54.38 |       0.06  |
|TweedieRegressor               |              0.36 |      0.43 | 54.88 |       0.01  |
|GammaRegressor                 |              0.35 |      0.43 | 55.15 |       0.02  |
|KNeighborsRegressor            |              0.35 |      0.42 | 55.20 |       0.01  |
|PassiveAggressiveRegressor     |              0.34 |      0.41 | 55.69 |       0.01  |
|RandomForestRegressor          |              0.34 |      0.41 | 55.74 |       0.03  |
|LGBMRegressor                  |              0.31 |      0.39 | 56.79 |       0.06  |
|LassoLars                      |              0.30 |      0.38 | 57.36 |       0.01  |
|BaggingRegressor               |              0.30 |      0.38 | 57.39 |       0.02  |
|HistGradientBoostingRegressor  |              0.30 |      0.38 | 57.54 |       0.18  |
|LinearSVR                      |              0.18 |      0.28 | 61.96 |       0.01  |
|OrthogonalMatchingPursuit      |              0.14 |      0.23 | 63.73 |       0.01  |
|XGBRegressor                   |              0.13 |      0.23 | 63.95 |       0.16  |
|SVR                            |              0.08 |      0.18 | 65.82 |       0.01  |
|NuSVR                          |              0.06 |      0.17 | 66.40 |       0.04  |
|DecisionTreeRegressor          |             -0.04 |      0.08 | 69.91 |       0.01  |
|QuantileRegressor              |             -0.13 |     -0.00 | 72.89 |       3.33  |
|DummyRegressor                 |             -0.14 |     -0.01 | 73.22 |       0.01  |
|RANSACRegressor                |             -0.22 |     -0.08 | 75.78 |       0.15  |
|ExtraTreeRegressor             |             -0.26 |     -0.12 | 77.01 |       0.01  |
|GaussianProcessRegressor       |             -0.80 |     -0.60 | 92.06 |       0.05  |
|MLPRegressor                   |             -1.13 |     -0.89 |100.05 |       0.41  |
|Lars                           |             -1.49 |     -1.20 |108.05 |       0.01  |
|KernelRidge                    |             -4.91 |     -4.24 |166.55 |       0.02  |

Now with this information, we can work on improving the accuracy of our model, by taking few of the top performing models and tuning them.


## Input Params for LazyRegressor and LazyClassifier

```python
LazyRegressor(verbose=0,ignore_warnings=True, custom_metric=None)
```

* Verbose - Default = 0 - Anything greater than 0, will just print output of each model, as the model is done training in the same line.
* Ignore_Warnings - Default = True - If set to False, will print warnings for each model.
* Custom_Metric - Default = None - If you want to use a custom metric, you can pass it here. It should be a function that takes in y_test and y_pred as input and returns a float value.

## Custom Metric

From the source of lazy predict

[The Line where Custom Metric is Used](https://github.com/shankarpandala/lazypredict/blob/dev/lazypredict/Supervised.py#L322)

```python
if self.custom_metric is not None:
    custom_metric = self.custom_metric(y_test, y_pred)
    CUSTOM_METRIC.append(custom_metric)
```

From here, we can tell that custom metric should be a list of functions, that take in y_test and y_pred as input and return a float value.


## Links

* [Lazy Predict Github](https://github.com/shankarpandala/lazypredict)
* [Lazy Predict Documentation](https://lazypredict.readthedocs.io/en/latest/)
* [Lazy Predict PyPi](https://pypi.org/project/lazypredict/)
* [Python Notebook](https://github.com/SuperSecureHuman/ML-Experiments/tree/main/LazyPredict)

## Conclusion
This is a amazing library, that can help you get started with your ML project, by giving you a quick overview of the models that you can use. It is also very useful for getting a quick overview of the performance of your model, by comparing it with other models. I hope you found this useful.