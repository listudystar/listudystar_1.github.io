
# DT-California房价预测

## （决策树预测+树可视化+模型参数遍历）


```python
import matplotlib.pyplot as plt
import pandas as pd
%matplotlib inline
```

### >> 1. 加载数据（去除NAN值）


```python
from sklearn.datasets.california_housing import fetch_california_housing # 内部房价预测数据集
housing = fetch_california_housing()
print(housing.DESCR)
```

    .. _california_housing_dataset:
    
    California Housing dataset
    --------------------------
    
    **Data Set Characteristics:**
    
        :Number of Instances: 20640
    
        :Number of Attributes: 8 numeric, predictive attributes and the target
    
        :Attribute Information:
            - MedInc        median income in block
            - HouseAge      median house age in block
            - AveRooms      average number of rooms
            - AveBedrms     average number of bedrooms
            - Population    block population
            - AveOccup      average house occupancy
            - Latitude      house block latitude
            - Longitude     house block longitude
    
        :Missing Attribute Values: None
    
    This dataset was obtained from the StatLib repository.
    http://lib.stat.cmu.edu/datasets/
    
    The target variable is the median house value for California districts.
    
    This dataset was derived from the 1990 U.S. census, using one row per census
    block group. A block group is the smallest geographical unit for which the U.S.
    Census Bureau publishes sample data (a block group typically has a population
    of 600 to 3,000 people).
    
    It can be downloaded/loaded using the
    :func:`sklearn.datasets.fetch_california_housing` function.
    
    .. topic:: References
    
        - Pace, R. Kelley and Ronald Barry, Sparse Spatial Autoregressions,
          Statistics and Probability Letters, 33 (1997) 291-297
    
    


```python
data = housing.data
data.shape
```




    (20640, 8)




```python
type(data) # 2D矩阵结构
```




    numpy.ndarray



### >> 2. 构建树进行训练


```python
from sklearn import tree
dtr = tree.DecisionTreeRegressor(max_depth = 2) # 实例化树模型（树深度=2）
dtr.fit(housing.data[:, [6, 7]], housing.target) # 用两列特征进行树训练【注意】训练数据为array结构
```




    DecisionTreeRegressor(criterion='mse', max_depth=2, max_features=None,
                          max_leaf_nodes=None, min_impurity_decrease=0.0,
                          min_impurity_split=None, min_samples_leaf=1,
                          min_samples_split=2, min_weight_fraction_leaf=0.0,
                          presort=False, random_state=None, splitter='best')



##### 输出构建的树的默认参数

### >> 3. 树可视化


```python
#要可视化显示 首先需要安装 graphviz   http://www.graphviz.org/Download..php
import os
os.environ["PATH"] += os.pathsep + 'C:/Program Files (x86)/Graphviz2.38/bin/' # 安装好graphviz 软件后，需要将环境变量更新

dot_data = \
    tree.export_graphviz(
        dtr, # 【改】决策树对象
        out_file = None,
        feature_names = housing.feature_names[6:8], # 【改】特征名改
        filled = True,
        impurity = False,
        rounded = True
    )
```


```python
import pydotplus
graph = pydotplus.graph_from_dot_data(dot_data)  # 【改】特征名改
graph.get_nodes()[7].set_fillcolor("#FFF2DD")
from IPython.display import Image
Image(graph.create_png())
```




![image](https://github.com/listudystar/listudystar.github.io/raw/master/_posts/20190725_1.png)




```python
graph.write_png("dtr_white_background.png") # 输出“树”png图片
```




    True



### >> 4. 数据集划分（用了8列特征）


```python
from sklearn.model_selection import train_test_split
data_train, data_test, target_train, target_test = \
    train_test_split(housing.data, housing.target, test_size = 0.1, random_state = 42)
dtr = tree.DecisionTreeRegressor(random_state = 42)
dtr.fit(data_train, target_train)

dtr.score(data_test, target_test)
```




    0.637355881715626



##### 1百万数据量都可以用 SKlearn

## 树模型参数:（树太大会造成过拟合）

-  1. criterion  gini  or  entropy（熵）

-  2. splitter  best or random 前者是在所有特征中找最好的切分点 后者是在部分特征中（数据量大的时候）

-  3. max_features  None（所有），log2，sqrt，N  特征小于50的时候一般使用所有的

-  4. 【重要】max_depth  数据少或者特征少的时候可以不管这个值，如果模型样本量多，特征也多的情况下，可以尝试限制下【预剪枝核心】。

-  5. 【重要】min_samples_split  如果某节点的样本数少于min_samples_split，则不会继续再尝试选择最优特征来进行划分如果样本量不大，不需要管这个值。如果样本量数量级非常大，则推荐增大这个值。【叶子节点分裂情况】

-  6. min_samples_leaf  这个值限制了叶子节点最少的样本数，如果某叶子节点数目小于样本数，则会和兄弟节点一起被剪枝，如果样本量不大，不需要管这个值，大些如10W可是尝试下5

-  7. min_weight_fraction_leaf 这个值限制了叶子节点所有样本权重和的最小值，如果小于这个值，则会和兄弟节点一起被剪枝默认是0，就是不考虑权重问题。一般来说，如果我们有较多样本有缺失值，或者分类树样本的分布类别偏差很大，就会引入样本权重，这时我们就要注意这个值了。

-  8. max_leaf_nodes 通过限制最大叶子节点数，可以防止过拟合，默认是"None”，即不限制最大的叶子节点数。如果加了限制，算法会建立在最大叶子节点数内最优的决策树。如果特征不多，可以不考虑这个值，但是如果特征分成多的话，可以加以限制具体的值可以通过交叉验证得到。

-  9. class_weight 指定样本各类别的的权重，主要是为了防止训练集某些类别的样本过多导致训练的决策树过于偏向这些类别。这里可以自己指定各个样本的权重如果使用“balanced”，则算法会自己计算权重，样本量少的类别所对应的样本权重会高。

- 10. min_impurity_split 这个值限制了决策树的增长，如果某节点的不纯度(基尼系数，信息增益，均方差，绝对差)小于这个阈值则该节点不再生成子节点。即为叶子节点 。
- 11. n_estimators:要建立树的个数【集成算法中应用】

### >> 5. 模型参数遍历（参数组合）【利用交叉验证+参数组合的方式遍历得到参数最优组合】


```python
from sklearn.model_selection import GridSearchCV
tree_param_grid = { 'max_depth': list((2,3,4,5,6,7)),'min_samples_split':list((2,3,4,5,6,7))} # 字典格式
grid = GridSearchCV(tree.DecisionTreeRegressor(),param_grid=tree_param_grid, cv=5) 
# RandomForestRegressor()表示算法；param_grid=tree_param_grid表示遍历参数形成的字典；cv表示5次交叉验证
grid.fit(data_train, target_train) # 训练
grid.best_params_, grid.best_score_ # 得出参数选择的组合结果
```




    ({'max_depth': 7, 'min_samples_split': 6}, 0.6668613643115991)




```python
dtr = tree.DecisionTreeRegressor(max_depth = 7,min_samples_split = 3, random_state = 42)
dtr.fit(data_train, target_train)
dtr.score(data_test, target_test)
```




    0.6523360526754658



##### 经过参数组合+ 交叉验证获得的一组DT参数，对于结果有提升


```python
pd.Series(dtr.feature_importances_, index = housing.feature_names).sort_values(ascending = False)
```




    MedInc        0.703292
    AveOccup      0.140363
    HouseAge      0.046535
    Latitude      0.040968
    AveRooms      0.033748
    Longitude     0.025317
    Population    0.008053
    AveBedrms     0.001722
    dtype: float64



##### 获得重要性指标


```python

```
