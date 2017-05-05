
# Feature Selection

<p>First, I explored the data and I decided to include all features since they were not such large number and all contained relevant information. Some features contained a large number of missing values and I did not include them in my analysis.<p> 

These were:

```loan_advances```
```restricted_stock_deferred```
```director_fees```

I used a min max scaler because I included features with very different scales, as the case of salary and from_messages, this way it is 
easier to compare them. 

I created one new feature: <br />
```sent_recieved_ratio```:  ratio of from messages to received messages.

The rationale behind this is that I thought persons of interest might send fewer messages related to the number of messages received.

Additionally, I also created the following three features: <br />
* ```poi_ratio_messages```:  ratio of sent and received messages to and from poi relative to total number of messages sent and received. <br /> 
* ```poi_from_ratio_messages```:  ratio of poi received messages to total messages received. <br />  
* ```poi_to_ratio_messages```: ratio of sent messages to poi relative to all messages sent. <br /> 

The rationale for poi related messages is that there might be more volume of e-mails sent among pois and this should also be relative to 
the number of e-mails received. I decided not to include this last three features and also from_poi_to_this_person and from_this_person_to_poi because of possible concerns with data leakage. However, I kept shared_receipt_with_poi.

For feature selection I first used ```SelectKBest``` on all my classifiers to see which features would be better to use in my algorithm. 
```SelectKBest``` looks at the differences in distribution of features between classes. I used the ```f_classif``` criteria to determine the importance of my features which uses an ANOVA F-value between label and features for classification tasks and chooses which combination of features has the best performance on the classifier.

The features selected and scores resulting from ```SelectKBest``` with k=9 were:

Selected Features <pre>['salary', 'bonus', 'long_term_incentive', 'deferred_income', 'total_payments', 'exercised_stock_options', 
'restricted_stock', 'total_stock_value', 'shared_receipt_with_poi'] </pre>

Feature Scores <pre>[18.289684043404513, 20.792252047181538, 9.9221860131898385, 11.458476579280697, 8.7727777300916809, 
24.815079733218194, 9.212810621977086, 24.182898678566872, 8.5894207316823774] </pre>

The feature I constructed returned the following score and p-value:
<pre>
Selected Features, Scores, P-Values
'sent_received_ratio', '0.12', '0.734'
</pre>

I also used principal component analysis and tested several number of components using ```GridSearchCV``` to see what number 
(of number of components) resulted in a best fit for my classifier. The number of components chosen by ```GridSearchCV``` equaled 3. The performance of my chosen classifier with the different number of components tested was the following:

<pre>
n_components=1
Accuracy: 0.87207       Precision: 0.54850      Recall: 0.22900 F1: 0.32310     
n_components=2
Accuracy: 0.85320       Precision: 0.42417      Recall: 0.28250 F1: 0.33914     
n_components=4
Accuracy: 0.84667       Precision: 0.41821      Recall: 0.38350 F1: 0.40010     
n_components=5
Accuracy: 0.84513       Precision: 0.41294      Recall: 0.38300 F1: 0.39741     
</pre>
