# -*- coding: utf-8 -*-
"""
Created on Thu Jul  4 11:13:01 2024

@author: Administrator
"""
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import chi2

desktop_path = 'C:/Users/Administrator/OneDrive/桌面'

# 设置参数
window_size = 250  # 滚动窗口期
alpha = 0.99

# 导入历史收益数据
df = pd.read_excel(desktop_path + '/FIS/培训ppt/301.xlsx')

# 计算滚动VaR
df['returns'] = df['O32账户当日损益']/df['市值敞口']
returns = df['returns']

df['VaR_returns'] = df['VaR(风险价值)']/df['市值敞口']

# 这里为了验证FIS系统的VaR模型 所以用了系统数字 否则可以用历史模拟法自己计算VaR
rolling_VaR = -df['VaR_returns'][window_size-1:]
#rolling_VaR = -pd.Series(returns).rolling(window=window_size).quantile(1 - alpha).dropna()

# 计算超出VaR的次数
exceptions = returns[window_size-1:] < -rolling_VaR.values
N_exc = np.sum(exceptions)
T = len(rolling_VaR)

# 无偏性测试 (Kupiec's Proportion of Failures Test)
LR_POF = -2 * np.log(((alpha**(T-N_exc)) * ((1-alpha)**(N_exc))) / (((N_exc/T)**N_exc) * ((1-N_exc/T)**(T-N_exc))))
p_value_POF = 1 - chi2.cdf(LR_POF, 1)

# 独立性测试 (Christoffersen's Independence Test)
n00 = np.sum((exceptions[:-1] == 0) & (exceptions[1:] == 0))
n01 = np.sum((exceptions[:-1] == 0) & (exceptions[1:] == 1))
n10 = np.sum((exceptions[:-1] == 1) & (exceptions[1:] == 0))
n11 = np.sum((exceptions[:-1] == 1) & (exceptions[1:] == 1))

p01 = n01 / (n00 + n01)
p11 = n11 / (n10 + n11)
p = (n01 + n11) / (n00 + n01 + n10 + n11)

LR_Ind = -2 * np.log((((1-p01)**n00) * (p01**n01) * ((1-p11)**n10) * (p11**n11)) / (((1-p)**(n00 + n10)) * (p**(n01 + n11))))
p_value_Ind = 1 - chi2.cdf(LR_Ind, 1)

# 联合测试 (Kupiec's Proportion of Failures Test + Christoffersen's Independence Test)
LR_CC = LR_POF + LR_Ind
p_value_CC = 1 - chi2.cdf(LR_CC, 2)

# 结果输出
print(f'Rolling VaR (250-day window, {alpha*100:.0f}% confidence):')
print(f'POF Test Statistic: {LR_POF:.2f}, p-value: {p_value_POF:.2f}')
print(f'Independence Test Statistic: {LR_Ind:.2f}, p-value: {p_value_Ind:.2f}')
print(f'Combined Test Statistic: {LR_CC:.2f}, p-value: {p_value_CC:.2f}')

# 绘制滚动VaR和超出点
plt.figure(figsize=(14, 7))
plt.plot(returns, label='Returns', alpha=0.75)
plt.plot(rolling_VaR.index, -rolling_VaR, color='red', linestyle='dashed', linewidth=2, label=f'Rolling VaR ({alpha*100:.0f}%)')
plt.scatter(rolling_VaR.index[exceptions], returns[window_size-1:][exceptions], color='red', label='Exceptions')
plt.title('Returns with Rolling VaR (250-day window)')
plt.xlabel('Time')
plt.ylabel('Returns')
plt.legend()
plt.grid(True)
plt.show()
