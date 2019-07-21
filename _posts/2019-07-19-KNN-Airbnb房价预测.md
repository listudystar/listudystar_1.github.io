
# KNN-Airbnb房价预测

### >> 1. 按规定列进行数据表读取


```python
import pandas as pd
features = ['accommodates','bedrooms','bathrooms','beds','price','minimum_nights','maximum_nights','number_of_reviews']
dc_listings  = pd.read_csv('listings.csv')
dc_listings = dc_listings[features]
```


```python
print(dc_listings.shape)
dc_listings.head()
```

    (3723, 8)
    




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
      <th>accommodates</th>
      <th>bedrooms</th>
      <th>bathrooms</th>
      <th>beds</th>
      <th>price</th>
      <th>minimum_nights</th>
      <th>maximum_nights</th>
      <th>number_of_reviews</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>4</td>
      <td>1.0</td>
      <td>1.0</td>
      <td>2.0</td>
      <td>$160.00</td>
      <td>1</td>
      <td>1125</td>
      <td>0</td>
    </tr>
    <tr>
      <th>1</th>
      <td>6</td>
      <td>3.0</td>
      <td>3.0</td>
      <td>3.0</td>
      <td>$350.00</td>
      <td>2</td>
      <td>30</td>
      <td>65</td>
    </tr>
    <tr>
      <th>2</th>
      <td>1</td>
      <td>1.0</td>
      <td>2.0</td>
      <td>1.0</td>
      <td>$50.00</td>
      <td>2</td>
      <td>1125</td>
      <td>1</td>
    </tr>
    <tr>
      <th>3</th>
      <td>2</td>
      <td>1.0</td>
      <td>1.0</td>
      <td>1.0</td>
      <td>$95.00</td>
      <td>1</td>
      <td>1125</td>
      <td>0</td>
    </tr>
    <tr>
      <th>4</th>
      <td>4</td>
      <td>1.0</td>
      <td>1.0</td>
      <td>1.0</td>
      <td>$50.00</td>
      <td>7</td>
      <td>1125</td>
      <td>0</td>
    </tr>
  </tbody>
</table>
</div>



### >> 2. 预处理【去除NAN值，某列格式化转换，标准化+归一化（因为要利用多特征，某些特征数值过大），洗牌，拆分】


```python
dc_listings = dc_listings.dropna() #去除NAN值
dc_listings.shape
```




    (3671, 8)




```python
dc_listings['price'] = dc_listings.price.str.replace("\$|,",'').astype(float) #转换price列中带$符号的字符串为float
dc_listings.head()
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
      <th>accommodates</th>
      <th>bedrooms</th>
      <th>bathrooms</th>
      <th>beds</th>
      <th>price</th>
      <th>minimum_nights</th>
      <th>maximum_nights</th>
      <th>number_of_reviews</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>4</td>
      <td>1.0</td>
      <td>1.0</td>
      <td>2.0</td>
      <td>160.0</td>
      <td>1</td>
      <td>1125</td>
      <td>0</td>
    </tr>
    <tr>
      <th>1</th>
      <td>6</td>
      <td>3.0</td>
      <td>3.0</td>
      <td>3.0</td>
      <td>350.0</td>
      <td>2</td>
      <td>30</td>
      <td>65</td>
    </tr>
    <tr>
      <th>2</th>
      <td>1</td>
      <td>1.0</td>
      <td>2.0</td>
      <td>1.0</td>
      <td>50.0</td>
      <td>2</td>
      <td>1125</td>
      <td>1</td>
    </tr>
    <tr>
      <th>3</th>
      <td>2</td>
      <td>1.0</td>
      <td>1.0</td>
      <td>1.0</td>
      <td>95.0</td>
      <td>1</td>
      <td>1125</td>
      <td>0</td>
    </tr>
    <tr>
      <th>4</th>
      <td>4</td>
      <td>1.0</td>
      <td>1.0</td>
      <td>1.0</td>
      <td>50.0</td>
      <td>7</td>
      <td>1125</td>
      <td>0</td>
    </tr>
  </tbody>
</table>
</div>




```python
from sklearn.preprocessing import StandardScaler
dc_listings[features] = StandardScaler().fit_transform(dc_listings[features]) #标准化
#normalized_listings = dc_listings
#print(dc_listings.shape)
#normalized_listings.head()
dc_listings.head()
```

    d:\ProgramData\Anaconda3\lib\site-packages\sklearn\preprocessing\data.py:645: DataConversionWarning: Data with input dtype int64, float64 were all converted to float64 by StandardScaler.
      return self.partial_fit(X, y)
    d:\ProgramData\Anaconda3\lib\site-packages\sklearn\base.py:464: DataConversionWarning: Data with input dtype int64, float64 were all converted to float64 by StandardScaler.
      return self.fit(X, **fit_params).transform(X)
    




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
      <th>accommodates</th>
      <th>bedrooms</th>
      <th>bathrooms</th>
      <th>beds</th>
      <th>price</th>
      <th>minimum_nights</th>
      <th>maximum_nights</th>
      <th>number_of_reviews</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>0.401420</td>
      <td>-0.249501</td>
      <td>-0.439211</td>
      <td>0.297386</td>
      <td>0.081119</td>
      <td>-0.341421</td>
      <td>-0.016575</td>
      <td>-0.516779</td>
    </tr>
    <tr>
      <th>1</th>
      <td>1.399466</td>
      <td>2.129508</td>
      <td>2.969551</td>
      <td>1.141704</td>
      <td>1.462622</td>
      <td>-0.065047</td>
      <td>-0.016606</td>
      <td>1.706767</td>
    </tr>
    <tr>
      <th>2</th>
      <td>-1.095648</td>
      <td>-0.249501</td>
      <td>1.265170</td>
      <td>-0.546933</td>
      <td>-0.718699</td>
      <td>-0.065047</td>
      <td>-0.016575</td>
      <td>-0.482571</td>
    </tr>
    <tr>
      <th>3</th>
      <td>-0.596625</td>
      <td>-0.249501</td>
      <td>-0.439211</td>
      <td>-0.546933</td>
      <td>-0.391501</td>
      <td>-0.341421</td>
      <td>-0.016575</td>
      <td>-0.516779</td>
    </tr>
    <tr>
      <th>4</th>
      <td>0.401420</td>
      <td>-0.249501</td>
      <td>-0.439211</td>
      <td>-0.546933</td>
      <td>-0.718699</td>
      <td>1.316824</td>
      <td>-0.016575</td>
      <td>-0.516779</td>
    </tr>
  </tbody>
</table>
</div>




```python
dc_listings = dc_listings.sample(frac = 1, random_state = 0) #洗牌
```


```python
dc_listings.head()
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
      <th>accommodates</th>
      <th>bedrooms</th>
      <th>bathrooms</th>
      <th>beds</th>
      <th>price</th>
      <th>minimum_nights</th>
      <th>maximum_nights</th>
      <th>number_of_reviews</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>792</th>
      <td>0.900443</td>
      <td>0.940003</td>
      <td>-0.439211</td>
      <td>0.297386</td>
      <td>-0.064303</td>
      <td>-0.065047</td>
      <td>-0.016575</td>
      <td>-0.140486</td>
    </tr>
    <tr>
      <th>122</th>
      <td>-1.095648</td>
      <td>-0.249501</td>
      <td>-0.439211</td>
      <td>0.297386</td>
      <td>-0.653260</td>
      <td>-0.341421</td>
      <td>-0.016575</td>
      <td>-0.482571</td>
    </tr>
    <tr>
      <th>1678</th>
      <td>-0.596625</td>
      <td>-1.439006</td>
      <td>-0.439211</td>
      <td>-0.546933</td>
      <td>-0.151556</td>
      <td>0.487701</td>
      <td>-0.016575</td>
      <td>1.227850</td>
    </tr>
    <tr>
      <th>2651</th>
      <td>-0.596625</td>
      <td>-0.249501</td>
      <td>-0.439211</td>
      <td>-0.546933</td>
      <td>-0.507838</td>
      <td>7.120680</td>
      <td>-0.016596</td>
      <td>6.598570</td>
    </tr>
    <tr>
      <th>2365</th>
      <td>0.401420</td>
      <td>-0.249501</td>
      <td>-0.439211</td>
      <td>-0.546933</td>
      <td>-0.216995</td>
      <td>0.211327</td>
      <td>-0.016575</td>
      <td>-0.482571</td>
    </tr>
  </tbody>
</table>
</div>




```python
norm_train_df = normalized_listings.copy().iloc[0:2792] #数据集拆分
norm_test_df = normalized_listings.copy().iloc[2792:]
```

### >> 3. 利用SKLEARN进行多特征KNN回归分析


```python
from sklearn.neighbors import KNeighborsRegressor
knn = KNeighborsRegressor()
cols = ['accommodates','bedrooms','bathrooms','beds','minimum_nights','maximum_nights','number_of_reviews'] #所采用的特征
knn.fit(norm_train_df[cols], norm_train_df['price']) #训练
features_predictions = knn.predict(norm_test_df[cols]) #回归预测
```

### >> 4. RMSE验证


```python
from sklearn.metrics import mean_squared_error
test_mse = mean_squared_error(norm_test_df['price'], features_predictions)
test_rmse = test_mse ** (1/2)
test_rmse #最终检验结果
```




    0.8243115782561288




```python

```
