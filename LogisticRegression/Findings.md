### Feb 13th, 2026
Logistic regression doing very well with topic, very bad with emotion. I want to generate higher quality examples.
Our TA also advised us to try to automatically validate the emotions to ensure they are quality in the dataset.

Thought about allowing the model to use the multiple classes labled by the LLM in the custom dataset, however I found that the model is pretty good at predicting the primary emotion. In cases where the model messes up, the second emotion is not predicted.
For example, joy is often labeled as surprise. However, there is only one example in the dataset where this should be the case

### Feb 18th, 2026
After changing the scale one the joint dataset scaling to log, it is clear to see that there are diagonals which deviate from the "true" diagonal on the joint dataset confusion matrix. Since the labels are ordered by emotion, these are emerging from correct topic classifications, but incorrect emotion classifications.
There are strong horizontal bands in emotions which are frequently misclassified regardless of topic such as joy and anger.