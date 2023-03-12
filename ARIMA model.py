#!/usr/bin/env python
# coding: utf-8

# In[1]:


import yfinance as yf


# In[2]:


# 삼전 코드
samsung = yf.download('005930.KS',start='2022-01-01',end='2022-12-01')

samsung = samsung[['Close']]
samsung.reset_index(inplace=True)
samsung = samsung.rename(columns={'Close':'Price'})
samsung.head(3)


# In[3]:


samsung.plot(x='Date',y='Price',kind='line')


# In[4]:


from statsmodels.tsa.arima.model import ARIMA
import statsmodels.api as sm


# In[5]:


# ARIMA model 생성
model = ARIMA(samsung.Price.values, order=(2,1,2))
# order의 파라미터는 (AR,Difference,MA)
# AR : AR이 몇 번째 과거까지 바라보는지에 대한 파라미터
# DIfference : 차분
# MA : MA가 몇 번째 과거까지 바라보는지에 대한 파라미터

fit = model.fit()
# 생성 모델에 대한 summary 확인
fit.summary()


# In[6]:


import matplotlib.pyplot as plt
import pandas as pd

pred = pd.DataFrame(fit.predict())
pred[1:].plot()
samsung['Price'].plot()


# In[7]:


residuals = pd.DataFrame(fit.resid)
residuals[1:].plot()


# In[18]:


# 예측치
forecast = fit.forecast(steps=10)
forecast


# In[14]:


# 실측치
forecasts = yf.download('005930.KS',start='2022-12-01',end='2022-12-10')
print(forecasts['Close'].values)


# In[15]:


forecasts.plot(xlabel='Date',y='Close',kind='line')


# In[16]:


forecasts


# In[21]:


plt.plot(forecast)


# In[ ]:




