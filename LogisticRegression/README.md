# Linear Regression Baseline Model

## Install dependencies
1. (Optional) run `python -m venv venv` to create a venv. Then source it with `source venv/bin/activarte` (MacOS)
2. Install requirements with `pip install -r requirements.txt`

## Run a quick evaluation
Run `quick_run.py` to download the datasets, fit the models, and score them on accuracy quickly

## Run a more in depth evaluation
Run `advanced_run.py` to do a more in-depth evaluation and produce more metrics, including the confusion matricies.
On the first run do not load the existing models.

## References
https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.LinearRegression.html
https://scikit-learn.org/stable/auto_examples/applications/plot_topics_extraction_with_nmf_lda.html
https://note.nkmk.me/en/python-pandas-map-applymap-apply/
https://www.geeksforgeeks.org/pandas/add-column-names-to-dataframe-in-pandas/
https://stackoverflow.com/questions/56621252/negative-accuracy-score-in-regression-models-with-scikit-learn
https://imbalanced-learn.org/stable/user_guide.html#user-guide
https://woteq.com/calculating-precision-and-recall-in-python-with-scikit-learn/
https://datascience.stackexchange.com/questions/122056/logisticregression-loading-problem

AI Used to:
 - help explain the difference between LinearRegression and LogisticRegression for this case
    - Allowed under [course AI policy](https://ics.uci.edu/~smyth/courses/cs175/academic_integrity.html) "for explaining error messages, for debugging"
 - make confusion matrix graph prettier
 - explain how to improve class imbalance
