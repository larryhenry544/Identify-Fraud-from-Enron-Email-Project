# Udacity-Identify-Fraud-from-Enron-Email-Project

**Larry Henry Jr.**

## Identity Fraud From Enron Email

### Project Introduction
In 2000, Enron was one of the largest companies in the United States. By 2002, it had collapsed into bankruptcy due to widespread corporate fraud. In the resulting Federal investigation, there was a significant amount of typically confidential information entered into public record, including tens of thousands of emails and detailed financial data for top executives. In this project, you will play detective, and put your new skills to use by building a person of interest identifier based on financial and email data made public as a result of the Enron scandal. To assist you in your detective work, we've combined this data with a hand-generated list of persons of interest in the fraud case, which means individuals who were indicted, reached a settlement, or plea deal with the government, or testified in exchange for prosecution immunity.

### Project Overview
In this project, you will play detective, and put your machine learning skills to use by building an algorithm to identify Enron Employees who may have committed fraud based on the public Enron financial and email dataset.

![Corporate corpus: Volumes of e-mails that were sent and received in Enron’s headquarters in Houston, seen here in 2002, are still parsed and dissected by computer scientists and other researchers.](https://cdn.technologyreview.com/i/images/enronx299.jpg?sw=680)  
_Image Source: https://www.technologyreview.com/s/515801/the-immortal-life-of-the-enron-e-mails/_


#### Why this Project?
This project will teach you the end-to-end process of investigating data through a machine learning lens.

It will teach you how to extract/identify useful features that best represents your data, a few of the most commonly used machine learning algorithms today, and how to evaluate the performance of your machine learning algorithms.

#### What will I learn?
By the end of the project, you will be able to:

Deal with an imperfect, real-world dataset
Validate a machine learning result using test data
Evaluate a machine learning result using quantitative metrics
Create, select and transform features compare the performance of machine learning algorithms
Tune machine learning algorithms for maximum performance
Communicate your machine learning algorithm results clearly
Why is this Important to my Career?
Machine Learning is a first-class ticket to the most exciting careers in data analysis today.

As data sources proliferate along with the computing power to process them, going straight to the data is one of the most straightforward ways to quickly gain insights and make predictions.

Machine learning brings together computer science and statistics to harness that predictive power.


### Resources Needed
You should have python and sklearn running on your computer, as well as the starter code (both python scripts and the Enron dataset) that you downloaded as part of the first mini-project in the Intro to Machine Learning course. The starter code can be found in the final_project directory of the codebase that you downloaded for use with the mini-projects. Some relevant files:

`poi_id.py` : starter code for the POI identifier, you will write your analysis here

`final_project_dataset.pkl` : the dataset for the project, more details below

`tester.py` : when you turn in your analysis for evaluation by a Udacity evaluator, you will submit the algorithm, dataset and list of features that you use (these are created automatically in poi_id.py). The evaluator will then use this code to test your result, to make sure we see performance that’s similar to what you report. You don’t need to do anything with this code, but we provide it for transparency and for your reference.

`emails_by_address` : this directory contains many text files, each of which contains all the messages to or from a particular email address. It is for your reference, if you want to create more advanced features based on the details of the emails dataset.

### Steps to Success
We will provide you with starter code, that reads in the data, takes your features of choice, then puts them into a numpy array, which is the input form that most sklearn functions assume. Your job is to engineer the features, pick and tune an algorithm, test, and evaluate your identifier. Several of the mini-projects were designed with this final project in mind, so be on the lookout for ways to use the work you’ve already done.

The features in the data fall into three major types, namely financial features, email features and POI labels.

> __financial features__: ['salary', 'deferral_payments', 'total_payments', 'loan_advances', 'bonus', 'restricted_stock_deferred', 'deferred_income', 'total_stock_value', 'expenses', 'exercised_stock_options', 'other', 'long_term_incentive', 'restricted_stock', 'director_fees'] 
> > (all units are in US dollars)

> __email features__: ['to_messages', 'email_address', 'from_poi_to_this_person', 'from_messages', 'from_this_person_to_poi', 'shared_receipt_with_poi'] 
> > (units are generally number of emails messages; notable exception is ‘email_address’, which is a text string)

> __POI label__: [‘poi’] 
> > (boolean, represented as integer)

You are encouraged to make, transform or rescale new features from the starter features. If you do this, you should store the new feature to my_dataset, and if you use the new feature in the final algorithm, you should also add the feature name to my_feature_list, so your coach can access it during testing. For a concrete example of a new feature that you could add to the dataset, refer to the lesson on Feature Selection.
    