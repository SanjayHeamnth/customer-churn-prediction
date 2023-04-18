# CUSTOMER CHURN PREDICTION AND CUSTOMER CLUSTERING
## Predicting Customer Churn with Machine Learning Classification Algorithm

#

**NOTE:** for Customer Clustering, please check the repository [here](customer-clustering/). However, we analyse the dataset here.

<br/>

<p align="center"><img src="img/cust-churn-img.jpeg" width="480"></p>

<br/>

## About the project
*Churn* can be defined as customer who stop, discontinue, or unsubscribe to a service or business. On a business, maintaining a customer was an important thing to do, yet it could be really hard to do. One way to predict customer behaviour is to analyse customer based on data. In the era of Big Data and Machine Learning, we can learn about customer and analyse customer behaviour pattern to do a prediction.
By building a model, companies can predict customer who're at high risk of churn and find new strategy to maintain customers.

# 

### **Objective**
Exploring and Analyze data and try to answer some question such as:
- What feature shows correlation to Churn Rate?
- Does the churn rate on the dataset skewed?
- How different the behaviour of customer who churned and don't?

Model Building and Metric:
- Whats the best accuracy of the model?
- What model predict the best?

#

***check full jupyter notebook [here](customer_churn.ipynb)***

#


```

#

## TABLE OF CONTENT
- THE DATASET
- EXPLORATORY DATA ANALYSIS (EDA)
- MODEL RESULT AND CONCLUSION

#

## **THE DATASET**
Dataset sourced from IBM Telco Customer dataset, which uploaded on kaggle by BlastChar: https://www.kaggle.com/blastchar/telco-customer-churn

* dataset consist of 7043 row (customer) with 21 column (features) described as:
    - customerID        : unique id for each customer
    - gender            : gender of customer
    - SeniorCitizen     : whether the customer is a senior citizen (yes/no)
    - Partner           : whether the customer has a partner or not
    - Dependents        : whether the customer has a dependent or not 
    - tenure            : month count of customer has stayed on the company
    - PhoneService      : whether the customer has a phone service or not
    - MultipleLines     : whether the customer has multiple lines or not
    - InternetService   : customer's ISP (internet service provider)
    - OnlineSecurity    : whether the customer has online security or not
    - OnlineBackup      : whether the customer has online backup or not
    - DeviceProtection  : whether the customer has device protection or not
    - TechSupport       : whether the customer has tech support or not
    - StreamingTV       : whether the customer has streaming tv or not
    - StreamingMovies   : whether the customer has streaming movies or not
    - Contract          : contract term of customer
    - PaperlessBilling  : whether the customer has paperless billing or not
    - PaymentMethod     : customer's payment method
    - MonthlyCharges    : customer's amount of charges monthly
    - TotalCharges      : total amount of customer's charges
    - Churn             : whether the customer churned or not

#

## **EXPLORATORY DATA ANALYSIS (EDA)**
*check jupyter notebook for complete EDA and code*

<br/>

### **Churn vs Not Churn ratio**

In the model, Churn is the target of the classification.

<p align="center"><img src="plot/1.png" width=250px></p>

73,4 customer not churned. The dataset is skewed, but its normal since on a business, we expect there's more customer who stayed. However, this skewness can lead to false negatives. The skewness is handled with upsampling minorities(churn), so the ratio is balanced

<p align="center"><img src="plot/30.png" width=350px></p>

<br/>
<br/>
<br/>

### **Tenure, Monthly Charges, Total Charges**

tenure, MonthlyCharges, and TotalCharges are numerical feature on the data.

Analyse distribution:

<p align="center">
<img src="plot/2.png" width=350px>
<img src="plot/5.png" width=350px>
</p>
customer who churned tend to have tenure on 0-20

<br/>
<br/>

<p align="center">
<img src="plot/3.png" width=350px>
<img src="plot/6.png" width=350px>
</p>
customer who churned tend to have higher MonthlyCharges, while customer who not churned tend to have lower Monthly Charges

<br/>
<br/>

<p align="center">
<img src="plot/4.png" width=350px>
<img src="plot/7.png" width=350px>
</p>
there's no pattern on TotalCharges. both customer who churn and not churn tend to have lower TotalCharges between 0-2000. There's slightly higher density on higher TotalCharges on customer who not churn, but still mostly on lower TotalCharges.

<br/>
<br/>


**MonthlyCharges vs TotalCharges**

<p align="center"><img src="plot/8.png" width=400px></p>

there are linear relationship, where bigger the Monthly Charges, bigger the Total Charges, which was normal.
<br/>
<br/>

### **Categorical Features**
<br/>

- Gender

<p align="center"><img src="plot/9.png" width=400px></p>
<br/>

female and male customer count balanced and Churn ratio also similar, this mean there's no majority gender on customer and one gender don't tend to Churn.

- SeniorCitizen

<p align="center"><img src="plot/10.png" width=400px>
<br/></p>

most of customer wasn't a senior citizen. however, senior citizen has more churn rate then customer who are not a senior citizen.

- Partner

<p align="center"><img src="plot/11.png" width=400px>
<br/></p>

customer who has and doesn't has partner count is balanced. customer who don't has partner show slightly higher churn rate.

- Dependents

<p align="center"><img src="plot/12.png" width=400px></p>
<br/>

most customer doesn't have dependents. customer who has dependents shows lower churn rate the customer who don't

- PhoneService

<p align="center"><img src="plot/13.png" width=480px></p>
<br/>

majority of customer have PhoneService, there's no significant churn ratio deferences between customer who have Phone Service or not

- MultipleLines

<p align="center"><img src="plot/14.png" width=480px></p>
<br/>

out of 90,3% customer who have Phone Service, half of them used MultipleLines and others don't. There's also ni significant churn ratio differences between cutomer who used Multiple Lines and not

- InternetService

<p align="center"><img src="plot/15.png" width=480px></p>
<br/>

customer who used fiber optic shows more churn rate than the other two, with customer who didn't use internet service has the lowest churn rate

- OnlineSecurity

<p align="center"><img src="plot/16.png" width=480px></p>
<br/>

half of the customer don't used OnlineSecurity, and has the highest churn rate.

- OnlineBackup

<p align="center"><img src="plot/17.png" width=480px></p>
<br/>

- DeviceProtection

<p align="center"><img src="plot/18.png" width=480px></p>
<br/>

- TechSupport

<p align="center"><img src="plot/19.png" width=480px></p>
<br/>

half of customer has no tech support and 41,6% of customer with no tech support churned. this comparision is so big compared to customer with tech support with only 15,2& churn rate

- StreamingTV

<p align="center"><img src="plot/20.png" width=480px></p>
<br/>

- StreamingMovies

<p align="center"><img src="plot/21.png" width=480px></p>
<br/>

- Contract

<p align="center"><img src="plot/22.png" width=480px></p>
<br/>

month-to-month contract being a majority, and with the highest churn rate od 42,7% meanwhile one year contract has 11.3% churn rate and two year only 2,8% churn rate. this show loyal customer tend to make contract with longer period.


<p align="center"><img src="plot/23.png" width=500px></p>
<br/>

comparision:

<p align="center"><img src="plot/24.png" width=450px></p>
<br/>

- Paperless Billing


<p align="center"><img src="plot/25.png" width=480px></p>
<br/>

majority user used paperless billing, but have slightly more churn rate too.

- Payment Method

<p align="center"><img src="plot/26.png" width=350px></p>
<br/>

payment method distributed pretty well, electronic check slightly have bigger count.

<p align="center"><img src="plot/27.png" width=480px></p>
<br/>

yet being slightly larger count than other, electronic check has biggest churn rate of 45,3% while other has only under 20% churn rate.

<br/>
<br/>

### **Correlation Matrix**
From all feature listed, we clearly can see what features effecting churn rate and whats dont. for example, tenure seems correlated well with churn because there's slightly visible imbalance of tenure who churn vs tenure who don't. other example of correlated well with churn is senior citizen, where there're more churn ratio on senior citizen the customer who are not senior citizen. In other hand, feature like gender seems not correlated well with churn rate, there're no visbile difference in count of each gender who tend to churn. 

With this as assumption, we also can measure the correlation with correlation matrix.

**Correlation Heatmap for every feature:**

<p align="center"><img src="plot/28.png" width=500px></p>
<br/>

from the heatmap above, feature that highly correlated shown by darker red (positively highly correlated) or darker blue (negatively highly correlated). we can see tenure is highly correlated to totalcharges, churn, payment method, etc.

**Correlation Heatmap on every feature to Churn:**

<p align="center"><img src="plot/29.png" height=550px></p>
<br/>

we can see that the most positive correlated with churn is PaperlessBilling, MonthlyCharges, and SeniorCitizen. and the most negatively correlated with churn is Contract, tenure, and OnlineSecurity. we've predict some of this before on our data analysis by ploting. 

Based on Feature Correlation to Churn, I decide to not include really low correlation score to the model as it can categorized as noise. The model feed with feature:

['Churn', 'Contract', 'tenure', 'OnlineSecurity', 'TechSupport', 'TotalCharges', 'OnlineBackup', 'PaperlessBilling', 'MonthlyCharges', 'DeviceProtection', 'Dependents', 'SeniorCitizen', 'Partner', 'PaymentMethod']

#

## **MODEL RESULT AND CONCLUSION**

The model was trained with train vs test data ratio of 70% train and 30% test.

I used 6 Machine Learning Algorithm to predict/classified customer as Churn or Not Churn:
- Random Forest
- Decision Tree
- XG Boost
- Adaboost
- Logistic Regression
- SVM

Model Accuracy Comparision:

                    model       acc
    5        Random Forest  0.889472
    1        Decision Tree  0.861922
    3             XG Boost  0.843555
    4             Adaboost  0.768121
    2                  SVM  0.759265
    0  Logistic Regression  0.755986

#

## CONCLUSION
We've implemented Churn Prediction with several machine learning algorithm with performance accuracy of 75-88%. With this model, we can improve business, esspecially to maintain customer from churn or leaving. 
