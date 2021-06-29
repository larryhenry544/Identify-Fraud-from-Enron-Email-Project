<h1 style="text-align: center;">Identify Fraud from Enron Email Project</h1>
<h4 style="text-align: center;"><strong>Larry Henry Jr.</strong></h4>

## Introduction
<blockquote>
    <p>Summarize for us the goal of this project and how machine learning is useful in trying to accomplish it. As part 
        of your answer, give some background on the dataset and how it can be used to answer the project question.</p>
</blockquote>

In 2000, Enron was one of the largest companies in the United States. By 2002, it had collapsed into bankruptcy due to 
widespread corporate fraud. In the resulting Federal investigation, there was a significant amount of typically 
confidential information entered into public record, including tens of thousands of emails and detailed financial data 
for top executives. In this project, you will play detective, and put your new skills to use by building a person of 
interest identifier based on financial and email data made public as a result of the Enron scandal. To assist you in 
your detective work, we've combined this data with a hand-generated list of persons of interest in the fraud case, which
means individuals who were indicted, reached a settlement, or plea deal with the government, or testified in exchange 
for prosecution immunity.

## Objective
The project's dataset has been combined with a hand-generated list of persons of interest (POI) in the fraud case, which
means individuals, who were indicted, reached a settlement or plea deal with the government, or testified in exchange 
for prosecution immunity. 

The dataset, before any transformations, contained 146 records consisting of 14 financial features (all units are in 
US dollars), 6 email features (units are generally number of emails messages; notable exception is `email_address`,
which is a text string), and 1 labeled feature (`poi`).

The goal of this project is to create a model that, using the optimal combination of the available features, can 
identify whether a person is a POI or not. Since the dataset contains financial and email information that is common 
among most corporations, it could potentially be used to help identify person of interests in similar situations in other
companies.

### Data Structures
```python2
__financial features__: ['salary', 'deferral_payments', 'total_payments', 'loan_advances', 'bonus', 
'restricted_stock_deferred', 'deferred_income', 'total_stock_value', 'expenses', 'exercised_stock_options', 'other', 
'long_term_incentive', 'restricted_stock', 'director_fees'] 
```
 - (all units are in US dollars)

```python2
__email features__: ['to_messages', 'email_address', 'from_poi_to_this_person', 'from_messages', 'from_this_person_to_poi', 'shared_receipt_with_poi'] 
```
 - (units are generally number of emails messages; notable exception is ‘email_address’, which is a text string)

```python2
__POI label__: [‘poi’] 
```
 - (boolean, represented as integer)

### Model Development Plan
Developing the classification model will consist of 6 steps that can be iterated over until an acceptable model 
performance is obtained. These steps are as follows. 

1. Exploratory Data Analysis on the dataset to understand the data and investigate informative features 
2. Select features likely to provide predictive power in the classification model 
3. Clean or remove erroneous and outlier data from the selected features 
4. Engineer/transform selected features into new features appropriate for classification modelling 
5. Test different classifiers and review performance 
6. Investigate new data sources that may exist to provide better model performance

## Data Exploration and Cleaning
<blockquote>
    <p>Were there any outliers in the data when you got it, and how did you handle those?</p>
    <blockquote>
        [relevant rubric items: “data exploration”, “outlier investigation”]
    </blockquote>
</blockquote>

Using exploratory data analysis (EDA) I was able to identify outliers, relationships and patterns in the dataset.

I initially identified and removed the following records:

* TOTAL: An extreme outlier that represents the sum of all the numerical data.
* THE TRAVEL AGENCY IN THE PARK: This record is obviously not an ENRON employee.
* LOCKHART EUGENE E: This record contained only nulls and is therefore not useful.

Next, after removing the above records, the dataset contained many missing values seen below:

Feature | Missing Value Count
---|:---:
salary  |  51
to_messages  |  60
deferral_payments  |  107
total_payments  |  21
loan_advances  |  142
bonus  |  64
email_address  |  35
restricted_stock_deferred  |  128
total_stock_value  |  20
shared_receipt_with_poi  |  60
long_term_incentive  |  80
exercised_stock_options  |  44
from_messages  |  60
other  |  53
from_poi_to_this_person  |  60
from_this_person_to_poi  |  60
poi  |  0
deferred_income  |  97
expenses  |  51
restricted_stock  |  36
director_fees  |  129

The number of POIs is 18 representing a 12.58% of the total number of individuals.

Based on the scatter plots, we can still observe visible outliers in most features. These outliers do represent a 
significant proportion of the executives of the company and cannot be removed. We can also observe that the Email 
features have a totally different scale than the financial features.

With further analysis of the dataset, we see the following.

* Total number of data points: 146
* Total number of POI (persons of interest):18
* Total number of non POI: 128
* Total number of features per person: 21

Percentage of persons who has email data =  60.1398601399 %

Percentage of existing data in data set by feature

* poi  =  100.0 %
* fraction_form_this_person_to_poi  =  100.0 %
* fraction_to_this_person_from_poi  =  100.0 %
* total_stock_value  =  87.4125874126 %
* total_payments  =  86.013986014 %
* restricted_stock  =  76.2237762238 %
* exercised_stock_options  =  70.6293706294 %
* expenses  =  65.7342657343 %
* salary  =  65.7342657343 %
* other  =  63.6363636364 %
* to_messages  =  60.1398601399 %
* from_poi_to_this_person  =  60.1398601399 %
* shared_receipt_with_poi  =  60.1398601399 %
* from_messages  =  60.1398601399 %
* from_this_person_to_poi  =  60.1398601399 %
* bonus  =  56.6433566434 %
* long_term_incentive  =  45.4545454545 %
* deferred_income  =  33.5664335664 %
* deferral_payments  =  26.5734265734 %
* restricted_stock_deferred  =  11.8881118881 %
* director_fees  =  11.1888111888 %
* loan_advances  =  2.0979020979 %

## Feature Selection and Engineering
<blockquote>
  <p>
      What features did you end up using in your POI identifier, and what selection process did you use to pick them? Did
      you have to do any scaling? Why or why not?
  </p>
  <p>
    As part of the assignment, you should attempt to engineer your own feature that does not come ready-made in the 
    dataset -- explain what feature you tried to make, and the rationale behind it. (You do not necessarily have to use 
    it in the final analysis, only engineer and test it.) In your feature selection step, if you used an algorithm like 
    a decision tree, please also give the feature importances of the features that you use, and if you used an automated
    feature selection function like SelectKBest, please report the feature scores and reasons for your choice of 
    parameter values.
  </p>
  <blockquote>
      <p>[relevant rubric items: “create new features”, “intelligently select features”, “properly scale features”\]</p>
  </blockquote>
</blockquote>

I ended up using all features including 2 new features. The first feature, `fraction_to_this_person_from_poi`, 
represents the ratio of the messages from POI to this person divided with all the messages sent to this person. The 
second feature, `fraction_form_this_person_to_poi`, is the ratio from this person to POI divided with all messages from 
this person.

```python2
fraction_to_this_person_from_poi = from_poi_to_this_person / from_messages

fraction_form_this_person_to_poi = from_this_person_to_poi / to_messages
```
Since it is rare not to have email data, or just data is not available to us, I will use 0.1 as default value for 
persons who don't have email data for new features.

Also, I had scaled features using MinMaxScaler to convert all the features to a uniform scale because the dataset 
contains a lot of financial data, which the difference in numbers can be large like the number itself. To note, not all 
algorithms require feature scaling. For example, the Decision Tree doesn't require scaling because it doesn't rely on 
the Euclidean distance between data points when making decisions.

Univariate feature selection works by selecting the best features based on univariate statistical tests. Therefore, 
after scaling, I have used PCA (Principal Component Analysis) for dimensionality reduction retaining 15 components in 
which I used SelectKBest to retain 10 components with best score. 

Due to our data being sparse, I decided to go with `chi2` as the scoring function which is recommended for sparse data. 
Using `chi2` will handle the data without making it dense, which scored the features in the following order.

Feature | Score
---|:---:
 exercised_stock_options  |  6.845509335034564
 total_stock_value  |  5.47661009928604
 bonus  |  5.120754137086806
 salary  |  3.0527867447897865
 long_term_incentive  |  2.538485033080887
 fraction_to_this_person_from_poi  |  1.6125366070078333
 restricted_stock  |  0.5895353494865793
 deferred_income  |  0.3400992184059575
 fraction_form_this_person_to_poi  |  0.08564435674031562
 
In the following, I used the `select_features__k` from `SelectKBest` to algorithmically find the best number of features, 
using the `chi2` score together with the `DecisionTreeClassifier` and the `Pipeline` functions as follows:
```python2
n_features = np.arange(1, len(features_list))

pipe = Pipeline([
    ('select_features', SelectKBest(chi2)),
    ('classify', tree.DecisionTreeClassifier())
])

param_grid = [{'select_features__k': n_features}]
```

Then, I used GridSearchCVto automate the process of finding the optimal number of features based on their F1 score as follows:
```python2
tree_clf= GridSearchCV(pipe, param_grid=param_grid, scoring='f1', cv = 5)
tree_clf.fit(features, labels);
```

The following was the result.
```python2
{'select_features__k': 3}
```

I used the same procedure with the `RandomForestClassifier` and had results that were similar. As a result, I will use 
the following 9 features to train the algorithms:

Feature | 
---|
 exercised_stock_options  |
 total_stock_value  |
 bonus  |
 salary  |
 long_term_incentive  |
 fraction_to_this_person_from_poi  |
 restricted_stock  |
 deferred_income  |
 fraction_form_this_person_to_poi  |

## Algorithm Selection 
> What algorithm did you end up using? What other one(s) did you try? How did model
performance differ between algorithms? 
> > [relevant rubric item: “pick an algorithm”]

Tuning parameters of algorithm is a process that uses all input parameters of the algorithm that will impact the model 
in order to enable the best performance. Of course, best should be defined by what is important to us. As a result, I 
tested four supervised algorithms and one unsupervised, `KMeans`, for accuracy, precision and recall, and calling the 
`cross_val_score` helper function to perform cross-validation using the suggested **Scikit-learn** parameters. In 
addition, I tested using a different number of features in order to identify what impacts the nummber of features had on
the results.

The following shows the mean score and the 95% confidence interval of the score estimate.

<table>
  <caption>The mean score and the 95% confidence interval of the score estimate:</caption>
  <tr>
    <td></td>
    <th scope="col">Accuracy</th>
    <th scope="col">F1</th>
    <th scope="col">Precision</th>
    <th scope="col">Recall</th>
  </tr>
  <tr style="text-align: left;">
    <th scope="row">GaussianNB</th>
    <td>0.86 (+/- 0.13)</td>
    <td>0.22 (+/- 0.49)</td>
    <td>0.31 (+/- 0.74)</td>
    <td>0.52 (+/- 0.31)</td>
  </tr>
  <tr style="text-align: left;">
    <th scope="row">DecisionTreeClassifier</th>
    <td> 0.75 (+/- 0.13)</td>
    <td>0.35 (+/- 0.53)</td>
    <td>0.37 (+/- 0.25)</td>
    <td>0.20 (+/- 0.49)</td>
  </tr>
  <tr style="text-align: left;">
    <th scope="row">RandomForestClassifier</th>
    <td>0.86 (+/- 0.07)</td>
    <td>0.19 (+/- 0.49)</td>
    <td>0.61 (+/- 0.76)</td>
    <td>0.17 (+/- 0.28)</td>
  </tr>
  <tr style="text-align: left;">
    <th scope="row">LogisticRegression</th>
    <td>0.88 (+/- 0.13)</td>
    <td>0.23 (+/- 0.31)</td>
    <td>0.30 (+/- 0.49</td>
    <td>0.30 (+/- 0.53)</td>
  </tr>
  <tr style="text-align: left;">
    <th scope="row">KMeans</th>
    <td>0.54 (+/- 0.30)</td>
    <td>0.17 (+/- 0.57)</td>
    <td>0.12 (+/- 0.25)</td>
    <td>0.52 (+/- 0.67)</td>
  </tr>
</table>

Based on each model's performance, `KMeans` performs the worst but with a relative high accuracy. On the other hand,
this model returns a low precision and recall score followed by the `LogisticRegression` model. The 
`RandomForestClassifier`model performed better with the highest accuracy score and a high precision. However, the model 
has sub-requirements recall. The last model, `GaussianNB`, returned a moderate accuracy score, but with a relatively 
good precision and recall score.

Furthermore, the `DecisionTreeClassifier` model has the highest balance with precision and recall, which is also 
reflected with the highest *F1* score. This model has a moderate accuracy score and actually already satisfies the 
project's minimum scoring requirements without any extra parameter tuning. As a result, I will use this algorithm as 
well as the `RandomForestClassifierfor` for further tuning. Meanwhile, I will try to further improve their performance. 

In conclusion, based on the results, accuracy does not correlate with precision and recall scores in this dataset as the
labels are imbalanced.


## Algorithm and Parameter Tuning
> What does it mean to tune the parameters of an algorithm, and what can happen if you
don’t do this well? 

> How did you tune the parameters of your particular algorithm? What
parameters did you tune? (Some algorithms do not have parameters that you need to
tune -- if this is the case for the one you picked, identify and briefly explain how you
would have done it for the model that was not your final choice or a different model that
does utilize parameter tuning, e.g. a decision tree classifier).
> > [relevant rubric items: “discuss parameter tuning”, “tune the algorithm”\] 
 
Tuning parameters of an algorithm is a process in which we optimize the parameters that impact the model in order to 
enable the algorithm to perform the best. Therefore, when properly tuned, one can optimize their algorithm to its best 
performance. Otherwise, failure in choosing the best parameters may lead to low prediction power such as low accuracy, 
low precision, etc. 

Hyper-parameters are parameters that are not directly absorbed within estimators. By tuning the parameters, we try to 
achieve the maximum score for the given model.

We have to be cautious during the process of parameter and algorithm tuning because overtuning one or more of the 
model's paramaters may lead to very **high variance / low bias** (i.e. overfitting) and undertuning in very **high bias 
/ low variance** (i.e. underfitting).

The `GridSearchCV` method, found on the 
[sklearn](https://scikit-learn.org/stable/modules/generated/sklearn.model_selection.GridSearchCV.html) website, 
exhaustively generates candidates from a grid of parameter values and provides a dictionary with the best performing 
ones.

I used `GridSearchCV` to calculate the best possible parameters for the `DecisionTreeClassifier` from the following grid
of possible choices:
```python2
'criterion': ['gini', 'entropy'],
'min_samples_split': [2, 4, 6, 8, 10, 20],
'max_depth': [None, 5, 10, 15, 20],
'max_features': [None, 'sqrt', 'log2', 'auto']
```

The `GridSearchCV` model gave me the following for the best parameter combination:
```python2
{'criterion': 'gini',
 'max_depth': None,
 'max_features': 'auto',
 'min_samples_split': 6}
```

For the DecisionTreeClassifier, I used the following parameter grid:
```python2
"n_estimators": [9, 18, 27, 36],
"max_depth": [None, 1, 5, 10, 15],
"min_samples_leaf": [1, 2, 4, 6]
```

As a result, these optimized parameters returned:
```python2
{'max_depth': None, 'min_samples_leaf': 2, 'n_estimators': 9}
```

The following parameters were used to fit the models that returned the following results:
<table>
  <tr>
    <td></td>
    <th scope="col">Accuracy</th>
    <th scope="col">F1</th>
    <th scope="col">Precision</th>
    <th scope="col">Recall</th>
  </tr>
  <tr style="text-align: left;">
    <th scope="row">DecisionTreeClassifier</th>
    <td>0.82 (+/- 0.17)</td>
    <td>0.21 (+/- 0.67)</td>
    <td>0.51 (+/- 0.58)</td>
    <td>0.30 (+/- 0.53)</td>
  </tr>
  <tr style="text-align: left;">
    <th scope="row">RandomForestClassifier</th>
    <td>0.87 (+/- 0.10)</td>
    <td>0.30 (+/- 0.32)</td>
    <td>0.33 (+/- 0.37)</td>
    <td>0.12 (+/- 0.29)</td>
  </tr>
</table>

In the end, after parameter tuning, the `RandomForestClassifier` model improved the recall score. However, the precision
score decreased. Meanwhile, the `DecisionTreeClassifier` model showed a significant improvement over the previous 
results and is the classifier of choice.

## Validation
> What is validation, and what’s a classic mistake you can make if you do it wrong? How
did you validate your analysis? 
> > [relevant rubric items: “discuss validation”, “validation
strategy”\]

Validation is a technique that assesses how the results of a statistical analysis will generalize to an
independent data set for your model. A classic mistakes I often make is **overfitting** a model; consequently, the 
overfit model performs well on training data. However, this will typically fail drastically when making predictions 
about new or unseen data. For this reason it is common practice when performing a (supervised) machine learning 
experiment to hold out part of the available data as a test set `X_test`, `y_test`.

When it comes to evaluating different parameters, there is still a risk of overfitting on the test set because the 
parameters can be tweaked until the estimator performs optimally. To solve this problem, we can apply validation to 
another part of the dataset. Then, training can proceed on the training set. Afterwards, evaluation can be applied on 
the _validation set_, and final evaluation can be done on the test set.

There's another, more efficient validation method called **cross-validation**. In the basic approach, called 
`k-fold CV`, the training set is split into _k_ smaller sets. The model is then trained using _k-1_ of the folds as 
training data and the resulting model is validated on the remaining 1 fold. The performance measure reported by k-fold 
cross-validation is the average of the values computed in the loop. This approach can be computationally expensive, but 
conservative.

Due to the small size of the dataset, cross-validation was used by calling the `cross_val_score` helper function from 
[Scikit-learn](https://scikit-learn.org/stable/modules/generated/sklearn.model_selection.cross_val_score.html) with 5 
kfolds. On the other hand, `StratifiedKFold` is recommended in the case of imbalanced classes in which there were higher
performance with cv=5 based on observations.


## Evaluation
> Give at least 2 evaluation metrics and your average performance for each of them. Explain an interpretation of your 
metrics that says something human-understandable about your algorithm’s performance. 
> > [relevant rubric item: “usage of evaluation metrics”\]    

Evaluation metrics I have used are accuracy_score, precision_score and recall_score.

accuracy_score (Accuracy classification score)  = number of items in class labeled correctly/all items in that class 

- This function compute subset accuracy: the set of predicted true_positive must exactly match the corresponding in 
  true_positive.

precision_score = true_positives/(true_positives+false_positives) 

- Percentage of correctly identified/classified POI, exactness or quality.  

recall_score = true_positives/(true_positives+false_negatives) 

- Recall measures classification of true positives over all cases that are actually positives, completeness or quantity.


I have used `DecisionTreeClassifier` and I got results below:

Accuracy = 0.82187

-  How close to actual (true) value we are.

Precision = 0.31060

- There is 31.06% of chance that my model will predict actual POI.

Recall = 0.27550

- There is 27.55% chance that my model will predict actual POI correctly.

In conclusion, these metrics, although above project specifications, still seem rather low and further work with feature
engineering may possibly help to increase the scores.