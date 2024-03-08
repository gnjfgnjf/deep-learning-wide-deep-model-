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


# 결측치 채우기
test_df['Book-Author'].fillna('Na', inplace=True)


# In[4]:


def label_age(df):
    df['fix_age'] = 0  # 기본값으로 0으로 설정
    df.loc[df['Age'] < 10, 'fix_age'] = 0
    df.loc[(df['Age'] < 20) & (df['Age'] >= 10), 'fix_age'] = 10
    df.loc[(df['Age'] < 30) & (df['Age'] >= 20), 'fix_age'] = 20
    df.loc[(df['Age'] < 40) & (df['Age'] >= 30), 'fix_age'] = 30
    df.loc[(df['Age'] < 50) & (df['Age'] >= 40), 'fix_age'] = 40
    df.loc[(df['Age'] < 60) & (df['Age'] >= 50), 'fix_age'] = 50
    df.loc[(df['Age'] < 70) & (df['Age'] >= 60), 'fix_age'] = 60
    df.loc[(df['Age'] < 80) & (df['Age'] >= 70), 'fix_age'] = 70
    df.loc[(df['Age'] < 90) & (df['Age'] >= 80), 'fix_age'] = 80
    df.loc[(df['Age'] < 100) & (df['Age'] >= 90), 'fix_age'] = 90
    df.loc[df['Age'] >= 100, 'fix_age'] = 100
    df['fix_age'] = df['fix_age'].astype(float)
    return df

train_df = label_age(train_df)
test_df = label_age(test_df)

# 'Age' 열 삭제
train_df.drop(columns=['Age'], inplace=True)
test_df.drop(columns=['Age'], inplace=True)


# In[5]:


def fill_missing_values_with_mean(df):
    # Publisher를 기준으로 그룹화하고, 평균 계산
    publisher_mean = df[df['Year-Of-Publication'] != -1].groupby('Publisher')['Year-Of-Publication'].mean()

    # -1 값인 행에 대해 Publisher 그룹의 평균 값으로 채움
    df.loc[df['Year-Of-Publication'] == -1, 'Year-Of-Publication'] = df[df['Year-Of-Publication'] == -1]['Publisher'].map(publisher_mean)
    return df

train_df = fill_missing_values_with_mean(train_df)
test_df = fill_missing_values_with_mean(test_df)


# In[6]:


def fill_nan_with_mean(df, column_name):
    mean_value = df[column_name].mean()
    df[column_name].fillna(mean_value, inplace=True)
    return df

test_df = fill_nan_with_mean(test_df, 'Year-Of-Publication')
train_df = fill_nan_with_mean(train_df, 'Year-Of-Publication')


# In[7]:


# 연속형, 카테고리형 변수 리스트 정의
numeric_features = ['fix_age', 'Year-Of-Publication']
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


# In[8]:


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
model.summary()


# In[9]:


# 모델 학습
X_train = X_train.astype('float32')
X_val = X_val.astype('float32')

model.fit([X_train, X_train], y_train, epochs=10, batch_size=32, validation_data=([X_val, X_val], y_val))


# In[10]:


id_column = test_df['ID']
test_df.drop('ID', axis=1, inplace=True)

# 데이터 타입 변환
X_test = test_df.to_numpy(dtype='float32')

# 모델 예측
predictions = model.predict([X_test, X_test])

# 예측 결과 test에 추가
test_df['Book-Rating'] = predictions.flatten()

test_df.insert(0, 'ID', id_column)


# In[11]:


test_df = test_df[['ID', 'Book-Rating']]


# In[12]:


test_df


# In[ ]:




