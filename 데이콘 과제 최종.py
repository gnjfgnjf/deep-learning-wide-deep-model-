#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import re
from sklearn.model_selection import GridSearchCV
from sklearn.preprocessing import LabelEncoder, StandardScaler, OneHotEncoder
from sklearn.model_selection import train_test_split
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.layers import Input, Dense, Concatenate
from sklearn.impute import SimpleImputer
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Dense, concatenate
from tensorflow.keras.optimizers import Adam


# In[2]:


train_df = pd.read_csv("C:/Users/gnjfg/Downloads/open (1)/train.csv")
test_df = pd.read_csv("C:/Users/gnjfg/Downloads/open (1)/test.csv")


# In[3]:


# 연속형, 카테고리형 변수 리스트 정의
numeric_features = ['Age', 'Year-Of-Publication']
categorical_features = ['User-ID', 'Book-ID', 'Location', 'Book-Title', 'Book-Author', 'Publisher']

# 연속형 변수 결측치 처리, 표준화
for feature in numeric_features:
    train_df[feature].fillna(train_df[feature].mean(), inplace=True)
    test_df[feature].fillna(test_df[feature].mean(), inplace=True)

    scaler = StandardScaler()
    train_df[feature] = scaler.fit_transform(train_df[[feature]])
    test_df[feature] = scaler.transform(test_df[[feature]])

# 카테고리형 변수 전처리: 상위 100개 카테고리만 유지, 나머지는 'other'
N = 100
for feature in categorical_features:
    top_categories = list(train_df[feature].value_counts().nlargest(N).index)
    
    train_df[feature] = train_df[feature].apply(lambda x: x if x in top_categories else 'Other')
    test_df[feature] = test_df[feature].apply(lambda x: x if x in top_categories else 'Other')
    
    encoder = LabelEncoder()
    train_df[feature] = encoder.fit_transform(train_df[feature])
    test_df[feature] = encoder.transform(test_df[feature])

X = train_df.drop(['Book-Rating', 'ID'], axis=1)
y = train_df['Book-Rating']

X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)


# In[4]:


# Deep 부분 구성
deep_inputs = Input(shape=(X_train.shape[1],))
deep_part = Dense(128, activation='relu')(deep_inputs)
deep_part = Dense(64, activation='relu')(deep_part)
deep_part = Dense(32, activation='relu')(deep_part)

# Wide 부분 구성
wide_inputs = Input(shape=(X_train.shape[1],))
wide_part = Dense(32, activation='linear')(wide_inputs)

# 결합
combined = concatenate([wide_part, deep_part])

# 최종 출력 레이어
output = Dense(1, activation='linear')(combined)

# 모델 구성
model = Model(inputs=[wide_inputs, deep_inputs], outputs=output)

# 모델 컴파일
model.compile(optimizer=Adam(), loss='mse')


# In[5]:


# 모델 학습
model.fit([X_train, X_train], y_train, epochs=10, batch_size=32, validation_data=([X_val, X_val], y_val))


# In[6]:


id_column = test_df['ID']
test_df.drop('ID', axis=1, inplace=True)

# 데이터 타입 변환
X_test = test_df.to_numpy(dtype='float32')

# 모델 예측
predictions = model.predict([X_test, X_test])

# 예측 결과 test에 추가
test_df['Book-Rating'] = predictions.flatten()

test_df.insert(0, 'ID', id_column)


# In[7]:


test_df = test_df[['ID', 'Book-Rating']]


# In[8]:


test_df

