
# NLP Assignment: Predicting the lables of the paragraph.

### Importing libraries


```python
%matplotlib inline

import pandas as pd
import numpy as np
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.multiclass import OneVsRestClassifier
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score
from sklearn.linear_model import LogisticRegression
```

### Reading files


```python
train_data = pd.read_csv('C:/Users/prashant bajetha/train_data.csv')
train_label = pd.read_csv('C:/Users/prashant bajetha/train_label.csv')
test_df = pd.read_csv('C:/Users/prashant bajetha/test_data.csv')
```


```python
train_data.head()
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>text</th>
      <th>id</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>Keep your gloves, hats, coats and jackets toge...</td>
      <td>122885</td>
    </tr>
    <tr>
      <th>1</th>
      <td>The Home Dynamix Serendipity Ivory 5 ft. 2 in....</td>
      <td>188958</td>
    </tr>
    <tr>
      <th>2</th>
      <td>The Bosch 18-Volt lithium-ion line of Cordless...</td>
      <td>146065</td>
    </tr>
    <tr>
      <th>3</th>
      <td>Restore your Porter-Cable sander or polisher t...</td>
      <td>165138</td>
    </tr>
    <tr>
      <th>4</th>
      <td>The SPIKECUBE Surge Suppressor from Tripp Lite...</td>
      <td>185565</td>
    </tr>
  </tbody>
</table>
</div>




```python
train_label.head()
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>id</th>
      <th>label</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>100003</td>
      <td>Shape</td>
    </tr>
    <tr>
      <th>1</th>
      <td>100004</td>
      <td>Voltage (volts)</td>
    </tr>
    <tr>
      <th>2</th>
      <td>100004</td>
      <td>Wattage (watts)</td>
    </tr>
    <tr>
      <th>3</th>
      <td>100006</td>
      <td>Wattage (watts)</td>
    </tr>
    <tr>
      <th>4</th>
      <td>100007</td>
      <td>ENERGY STAR Certified</td>
    </tr>
  </tbody>
</table>
</div>




```python
test_df.head()
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>text</th>
      <th>id</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>These machine screw nuts are designed to be us...</td>
      <td>114689</td>
    </tr>
    <tr>
      <th>1</th>
      <td>The M18 FUEL Drill/Driver is the Most Powerful...</td>
      <td>183172</td>
    </tr>
    <tr>
      <th>2</th>
      <td>Steel City 2-Gang 30 cu. in. Square Electrical...</td>
      <td>217304</td>
    </tr>
    <tr>
      <th>3</th>
      <td>Native Collection Plus has Shaw's SilentStep P...</td>
      <td>184115</td>
    </tr>
    <tr>
      <th>4</th>
      <td>Fasade decorative 4 ft. x 8 ft. vinyl wall pan...</td>
      <td>103786</td>
    </tr>
  </tbody>
</table>
</div>



# Making train_label more understandable

### This problem is a Multi label classification problem in which each instance(row) is labelled with one or more than one label. So, to better understand the data set, i performed below operation.....


```python
n_label = train_label.label.unique()
n_label
```




    array(['Shape', 'Voltage (volts)', 'Wattage (watts)',
           'ENERGY STAR Certified', 'Finish', 'Indoor/Outdoor',
           'Package Quantity', 'Features', 'Included', 'Hardware Included',
           'Color', 'Assembly Required', 'Tools Product Type',
           'Commercial / Residential', 'Flooring Product Type'], dtype=object)




```python
original_label = train_label.copy()      # Making a original copy of train_label for use.
train_label.drop('label',axis = 1,inplace = True)      # dropping the label column.
train_label = train_label.drop_duplicates(subset=['id'])  # Only keeping the unique id's.
labels = ['Shape','Voltage (volts)', 'Wattage (watts)','ENERGY STAR Certified','Finish','Indoor/Outdoor','Package Quantity',
          'Features','Included','Hardware Included','Color','Assembly Required','Tools Product Type','Commercial / Residential',
          'Flooring Product Type']
for col in labels:
    train_label[col] = 0           # coverting train_label to get each label as a column and initialising the value with 0.
train_label.shape
```




    (41569, 16)




```python
for index, row in original_label.iterrows():     # replacing 0 with 1 where the label for a particular id are given.
    ID = row['id']
    Label = row['label']
    train_label.loc[train_label['id']==ID,Label] = 1
train_label.head()

```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>id</th>
      <th>Shape</th>
      <th>Voltage (volts)</th>
      <th>Wattage (watts)</th>
      <th>ENERGY STAR Certified</th>
      <th>Finish</th>
      <th>Indoor/Outdoor</th>
      <th>Package Quantity</th>
      <th>Features</th>
      <th>Included</th>
      <th>Hardware Included</th>
      <th>Color</th>
      <th>Assembly Required</th>
      <th>Tools Product Type</th>
      <th>Commercial / Residential</th>
      <th>Flooring Product Type</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>100003</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>1</th>
      <td>100004</td>
      <td>0</td>
      <td>1</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>2</th>
      <td>100006</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>3</th>
      <td>100007</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>4</th>
      <td>100008</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>1</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
    </tr>
  </tbody>
</table>
</div>



### Counting the number of labels for each sample


```python
counts = []
categories = list(train_label.columns.values)
for i in categories:
    counts.append((i, train_label[i].sum()))
df_stats = pd.DataFrame(counts, columns=['category', 'number_of_samples'])
df_stats

```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>category</th>
      <th>number_of_samples</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>id</td>
      <td>7055062493</td>
    </tr>
    <tr>
      <th>1</th>
      <td>Shape</td>
      <td>1468</td>
    </tr>
    <tr>
      <th>2</th>
      <td>Voltage (volts)</td>
      <td>2485</td>
    </tr>
    <tr>
      <th>3</th>
      <td>Wattage (watts)</td>
      <td>1727</td>
    </tr>
    <tr>
      <th>4</th>
      <td>ENERGY STAR Certified</td>
      <td>2954</td>
    </tr>
    <tr>
      <th>5</th>
      <td>Finish</td>
      <td>1461</td>
    </tr>
    <tr>
      <th>6</th>
      <td>Indoor/Outdoor</td>
      <td>4303</td>
    </tr>
    <tr>
      <th>7</th>
      <td>Package Quantity</td>
      <td>2019</td>
    </tr>
    <tr>
      <th>8</th>
      <td>Features</td>
      <td>1783</td>
    </tr>
    <tr>
      <th>9</th>
      <td>Included</td>
      <td>2020</td>
    </tr>
    <tr>
      <th>10</th>
      <td>Hardware Included</td>
      <td>2319</td>
    </tr>
    <tr>
      <th>11</th>
      <td>Color</td>
      <td>1987</td>
    </tr>
    <tr>
      <th>12</th>
      <td>Assembly Required</td>
      <td>2028</td>
    </tr>
    <tr>
      <th>13</th>
      <td>Tools Product Type</td>
      <td>2093</td>
    </tr>
    <tr>
      <th>14</th>
      <td>Commercial / Residential</td>
      <td>3093</td>
    </tr>
    <tr>
      <th>15</th>
      <td>Flooring Product Type</td>
      <td>1829</td>
    </tr>
  </tbody>
</table>
</div>



# Data preprocessing


```python
#combining both dataframe to include text column 
train_data.sort_values(['id'], ascending=[True], inplace=True)
train_df = pd.merge(train_data, train_label, on='id')
train_df.head()
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Unnamed: 0</th>
      <th>text</th>
      <th>id</th>
      <th>Shape</th>
      <th>Voltage (volts)</th>
      <th>Wattage (watts)</th>
      <th>ENERGY STAR Certified</th>
      <th>Finish</th>
      <th>Indoor/Outdoor</th>
      <th>Package Quantity</th>
      <th>Features</th>
      <th>Included</th>
      <th>Hardware Included</th>
      <th>Color</th>
      <th>Assembly Required</th>
      <th>Tools Product Type</th>
      <th>Commercial / Residential</th>
      <th>Flooring Product Type</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>0</td>
      <td>Classic architecture meets contemporary design...</td>
      <td>100003</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>1</th>
      <td>1</td>
      <td>The Grape Solar 265-Watt Polycrystalline PV So...</td>
      <td>100004</td>
      <td>0</td>
      <td>1</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>2</th>
      <td>2</td>
      <td>Achieving delicious results is almost effortle...</td>
      <td>100006</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>3</th>
      <td>3</td>
      <td>The Quantum Adjustable 2-Light LED Black Emerg...</td>
      <td>100007</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>4</th>
      <td>4</td>
      <td>The Teks #10 x 1-1/2 in. Zinc-Plated Steel Was...</td>
      <td>100008</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>1</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
    </tr>
  </tbody>
</table>
</div>




```python
import re 
import nltk
from nltk.corpus import stopwords
def preprocessing(dataset):
    corpus = []
    for i in range(0,len(dataset)):
        clean_text = re.sub('[^a-zA-Z]', ' ', dataset['text'][i])   
        clean_text = clean_text.lower()     # converts all the letters in small letter
        clean_text = clean_text.split()
        lm = WordNetLemmatizer()
        clean_text = [lm.lemmatize(word) for word in clean_text if not word in set(stopwords.words('english'))]  # it is checking for any stopwords and then converting the word into its root.
        clean_text = ' '.join(clean_text)
        corpus.append(clean_text)
    return corpus 

train_clean_text = preprocessing(train_df)
test_clean_text = preprocessing(test_df)

# replacing the text column with clean
train_df['text'] = train_clean_text
test_df['text'] = test_clean_text

train_df.head()

```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Unnamed: 0</th>
      <th>text</th>
      <th>id</th>
      <th>Shape</th>
      <th>Voltage (volts)</th>
      <th>Wattage (watts)</th>
      <th>ENERGY STAR Certified</th>
      <th>Finish</th>
      <th>Indoor/Outdoor</th>
      <th>Package Quantity</th>
      <th>Features</th>
      <th>Included</th>
      <th>Hardware Included</th>
      <th>Color</th>
      <th>Assembly Required</th>
      <th>Tools Product Type</th>
      <th>Commercial / Residential</th>
      <th>Flooring Product Type</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>0</td>
      <td>classic architecture meet contemporary design ...</td>
      <td>100003</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>1</th>
      <td>1</td>
      <td>grape solar watt polycrystalline pv solar pane...</td>
      <td>100004</td>
      <td>0</td>
      <td>1</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>2</th>
      <td>2</td>
      <td>achieving delicious result almost effortless w...</td>
      <td>100006</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>3</th>
      <td>3</td>
      <td>quantum adjustable light led black emergency l...</td>
      <td>100007</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>4</th>
      <td>4</td>
      <td>teks x zinc plated steel washer head hex self ...</td>
      <td>100008</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>1</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
    </tr>
  </tbody>
</table>
</div>




```python
#Splitting training data to train the model on one part and test it for another part.
train, test = train_test_split(train_df, random_state=1, test_size=0.30, shuffle=True)
X_train = train.text
X_test = test.text
```

###  Pipeline is to help automate machine learning workflows. So we use pipeline to train different classifier. 
# Naive Bayes



```python
categories =  ['Indoor/Outdoor',
                      'Commercial / Residential',
                       'ENERGY STAR Certified',
                       'Hardware Included',
                       'Package Quantity',
                       'Flooring Product Type',
                       'Color',
                       'Tools Product Type',
                       'Included',
                       'Voltage (volts)',
                       'Assembly Required',
                       'Features',
                       'Wattage (watts)',
                       'Finish',
                       'Shape']
NB_pipeline = Pipeline([
                ('tfidf', TfidfVectorizer()),
                ('clf', OneVsRestClassifier(MultinomialNB(
                    fit_prior=True, class_prior=None))),
            ])
for category in categories:
    print('... Processing {}'.format(category))
    # train the model on each label
    NB_pipeline.fit(X_train, train[category])
    # testing accuracy
    prediction = NB_pipeline.predict(X_test)
    print('Test accuracy is {}'.format(accuracy_score(test[category], prediction)))
```

    ... Processing Indoor/Outdoor
    Test accuracy is 0.8389864485606607
    ... Processing Commercial / Residential
    Test accuracy is 0.9113944350894074
    ... Processing ENERGY STAR Certified
    Test accuracy is 0.8854141608531794
    ... Processing Hardware Included
    Test accuracy is 0.8711410472295726
    ... Processing Package Quantity
    Test accuracy is 0.8943148103600352
    ... Processing Flooring Product Type
    Test accuracy is 0.9625531232459306
    ... Processing Color
    Test accuracy is 0.8923101595702029
    ... Processing Tools Product Type
    Test accuracy is 0.9031352738352979
    ... Processing Included
    Test accuracy is 0.8961590890866811
    ... Processing Voltage (volts)
    Test accuracy is 0.8935931360756956
    ... Processing Assembly Required
    Test accuracy is 0.9004089487611258
    ... Processing Features
    Test accuracy is 0.9220591772913158
    ... Processing Wattage (watts)
    Test accuracy is 0.9169272712693449
    ... Processing Finish
    Test accuracy is 0.9156442947638521
    ... Processing Shape
    Test accuracy is 0.9133990858792398
    

# Logistic Regression


```python
LogReg_pipeline = Pipeline([
                ('tfidf', TfidfVectorizer()),
                ('clf', OneVsRestClassifier(LogisticRegression(solver='sag'), n_jobs=1)),
            ])
for category in categories:
    print('... Processing {}'.format(category))
    # train the model using X_dtm & y
    LogReg_pipeline.fit(X_train, train[category])
    # compute the testing accuracy
    prediction = LogReg_pipeline.predict(X_test)
    print('Test accuracy is {}'.format(accuracy_score(test[category], prediction)))
```

    ... Processing Indoor/Outdoor
    Test accuracy is 0.935610616630583
    ... Processing Commercial / Residential
    Test accuracy is 0.9654398203832892
    ... Processing ENERGY STAR Certified
    Test accuracy is 0.9688076337102077
    ... Processing Hardware Included
    Test accuracy is 0.9469168470852377
    ... Processing Package Quantity
    Test accuracy is 0.9611899607088445
    ... Processing Flooring Product Type
    Test accuracy is 0.9898965600192446
    ... Processing Color
    Test accuracy is 0.9482800096223238
    ... Processing Tools Product Type
    Test accuracy is 0.9780290273434368
    ... Processing Included
    Test accuracy is 0.9655200064148826
    ... Processing Voltage (volts)
    Test accuracy is 0.9682463314890546
    ... Processing Assembly Required
    Test accuracy is 0.9695293079945474
    ... Processing Features
    Test accuracy is 0.9828401892390346
    ... Processing Wattage (watts)
    Test accuracy is 0.9669633549835619
    ... Processing Finish
    Test accuracy is 0.9684868895838344
    ... Processing Shape
    Test accuracy is 0.9610295886456579
    

### Logistic Regression here predicts the output with better test accuracy. So final predictions are made with Logistic regression

# Predicting probabilites


```python
#predicting probabilities
train_set = train_df.text
test_set = test_df.text
#making empty dataframe
submission = pd.DataFrame()
submission['id'] = test_df.id
for category in categories:
    #fitting the model to entire dataset.
    LogReg_pipeline.fit(train_set,train_df[category])           
    pred_prob = LogReg_pipeline.predict_proba(test_set)
    submission[category] = pred_prob[:,1]
    
submission.head()
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>id</th>
      <th>Indoor/Outdoor</th>
      <th>Commercial / Residential</th>
      <th>ENERGY STAR Certified</th>
      <th>Hardware Included</th>
      <th>Package Quantity</th>
      <th>Flooring Product Type</th>
      <th>Color</th>
      <th>Tools Product Type</th>
      <th>Included</th>
      <th>Voltage (volts)</th>
      <th>Assembly Required</th>
      <th>Features</th>
      <th>Wattage (watts)</th>
      <th>Finish</th>
      <th>Shape</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>114689</td>
      <td>0.497</td>
      <td>0.008</td>
      <td>0.004</td>
      <td>0.005</td>
      <td>0.995</td>
      <td>0.004</td>
      <td>0.006</td>
      <td>0.009</td>
      <td>0.011</td>
      <td>0.007</td>
      <td>0.002</td>
      <td>0.003</td>
      <td>0.003</td>
      <td>0.610</td>
      <td>0.009</td>
    </tr>
    <tr>
      <th>1</th>
      <td>183172</td>
      <td>0.008</td>
      <td>0.009</td>
      <td>0.011</td>
      <td>0.009</td>
      <td>0.006</td>
      <td>0.004</td>
      <td>0.015</td>
      <td>0.997</td>
      <td>0.008</td>
      <td>0.952</td>
      <td>0.006</td>
      <td>0.005</td>
      <td>0.007</td>
      <td>0.004</td>
      <td>0.002</td>
    </tr>
    <tr>
      <th>2</th>
      <td>217304</td>
      <td>0.024</td>
      <td>0.002</td>
      <td>0.005</td>
      <td>0.015</td>
      <td>0.913</td>
      <td>0.003</td>
      <td>0.024</td>
      <td>0.039</td>
      <td>0.077</td>
      <td>0.047</td>
      <td>0.030</td>
      <td>0.005</td>
      <td>0.021</td>
      <td>0.035</td>
      <td>0.026</td>
    </tr>
    <tr>
      <th>3</th>
      <td>184115</td>
      <td>0.893</td>
      <td>0.960</td>
      <td>0.025</td>
      <td>0.032</td>
      <td>0.013</td>
      <td>0.965</td>
      <td>0.043</td>
      <td>0.005</td>
      <td>0.026</td>
      <td>0.004</td>
      <td>0.008</td>
      <td>0.013</td>
      <td>0.016</td>
      <td>0.015</td>
      <td>0.015</td>
    </tr>
    <tr>
      <th>4</th>
      <td>103786</td>
      <td>0.203</td>
      <td>0.156</td>
      <td>0.019</td>
      <td>0.122</td>
      <td>0.002</td>
      <td>0.030</td>
      <td>0.393</td>
      <td>0.000</td>
      <td>0.057</td>
      <td>0.014</td>
      <td>0.020</td>
      <td>0.053</td>
      <td>0.036</td>
      <td>0.033</td>
      <td>0.031</td>
    </tr>
  </tbody>
</table>
</div>


