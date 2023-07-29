#!/usr/bin/env python
# coding: utf-8

# In[2]:


# get relevant Python libraries

import numpy  as np
import pandas as pd

import statsmodels.formula.api as smf
import scipy.stats as stats

import linearmodels.iv as iv

import wooldridge as woo


# In[3]:


# read datafile
alcohol= pd.read_csv('./alcohol.csv')


# In[4]:


#summary statistics
print(alcohol.info())
print(alcohol.describe())


# In[6]:


alcohol[["employ", "abuse"]].describe()


# In[7]:


## Fraction of employed
print(alcohol.employ.value_counts())


# In[8]:


## Fraction of abuse
print(alcohol.abuse.value_counts())


# In[ ]:


## Answer 1/1: Fraction of employed persons is 88.22% and fraction of abused persons is 88.48%.


# In[9]:


# OLS using the statsmodels library
olsmod = smf.ols(formula='employ ~ abuse + age + agesq + educ + educsq + married + famsize + white + northeast + midwest + 
                 south + centcity + outercity + qrt1 + qrt2 + qrt3',data=alcohol)

#
# OLS estimate
olsres = olsmod.fit(cov_type='HC1')
olsres.summary()


# In[ ]:


## Answer 1/2: R square and adjusted R square is very low, only 6.9% of the variation is explained by the model. 
## High F-statistic value with p value of F statistic is almost 0.
## Almost all variables are statistically significant except family size, location and quarter 2,3.


# In[10]:


# IV regression using GMM
#    this assumes a robust covariance matrix
#
gmrmod = iv.IVGMM.from_formula(formula='employ ~ 1 + age + agesq + educ + educsq + married + famsize + white + northeast + 
                               midwest + south + centcity + outercity + qrt1 + qrt2 + qrt3 + [abuse ~ mothalc + fathalc]',weight_type='robust',data=alcohol)
gmrres = gmrmod.fit(cov_type='robust')

print(gmrres)


# In[ ]:


## Answer 1/3: R square and adjusted R square is negative. 
## High F-statistic value with p value of F statistic is almost 0 which suggests that the model fits the data.
## Almost all variables are statistically significant except family size, location and quarter 3.
## Endogenous variable "abuse" has higher negative value (-0.3545) but it was only -0.0202 in the OLS regression.


# In[11]:


# testing for weak instruments                      -
# first stage equation
fsmod = smf.ols(formula='abuse ~ age + agesq + educ + educsq + married + famsize + white + northeast + midwest + south + 
                centcity + outercity + qrt1 + qrt2 + qrt3 + fathalc + mothalc',data=alcohol)
fsres = fsmod.fit()
fstab = pd.DataFrame({'b'   : round(fsres.params,4),
                      'se'  : round(fsres.bse, 4),
                      't'   : round(fsres.tvalues, 2),
                      'pval': round(fsres.pvalues,4)})
print()
print('First stage regression')
print(fstab)


# In[27]:


import sys


# In[12]:


# testing for weak instruments                      -
# F-test for weak
wihyp  = ['fathalc=0', 'mothalc=0']
witest = fsres.f_test(wihyp)
wistat = witest.statistic
wipval = witest.pvalue

print()
print('Test for weak instruments')
print('F-stat : {}'.format(wistat))
print('p-value: {}'.format(wipval))


# In[ ]:


## Answer 1/4-1: F-stat > 22 not weak instruments.


# In[13]:


## Wooldridge's Hausman-Wu test statistics
# testing for endogenous                      -
# first stage equation
alcohol['abuserhat'] = fsres.resid


# In[14]:


# second stage equation (2SLS version 2)
tsls2mod = smf.ols(formula='employ ~ abuse + age + agesq + educ + educsq + married + famsize + white + northeast + midwest + 
                   south + centcity + outercity + qrt1 + qrt2 + qrt3 + abuserhat',data=alcohol)
tsls2res = tsls2mod.fit()
tsls2tab = pd.DataFrame({'b'   : round(tsls2res.params,4),
                      'se'  : round(tsls2res.bse, 4),
                      't'   : round(tsls2res.tvalues, 2),
                      'pval': round(tsls2res.pvalues,4)})
print()
print('Two stage regression - version 2')
print(tsls2tab)


# In[15]:


# test for endogeneity
endhyp = ['abuserhat=0']
endtest = tsls2res.f_test(endhyp)
wistat = witest.statistic
wipval = witest.pvalue

print()
print('Wooldridge-Hausman-Wu test for endogeneity')
print('F-stat : {}'.format(endtest.statistic))
print('p-value: {}'.format(endtest.pvalue))


# In[ ]:


## Answer 1/4-2: Endogeniety present at 98% level of significance.


# In[ ]:


# instrument validity      
# Sargan's J-stat test


# In[53]:


alcohol['tslsrhat'] = tsls2res.resid
sjmod = smf.ols(formula='tslsrhat ~ abuse + age + agesq + educ + educsq + married + famsize + white + northeast + midwest + 
                south + centcity + outercity + qrt1 + qrt2 + qrt3 + fathalc + mothalc',data=alcohol)
sjres = sjmod.fit()

sj_lm = sjres.nobs * sjres.rsquared
sj_pv = 1 - stats.chi2.cdf(x=sj_lm,df=1)

print()
print('Aux reg Rsqrd: {}'.format(round(sjres.rsquared,6)))
print('Sargan J-test: {}'.format(round(sj_lm,4)))
print('      p-value: {}'.format(round(sj_pv,4)))


# In[ ]:


## Answer 1/4-3: we fail to reject the null hypothesis which means that there is no correaltion between father's alcoholic 
## and mother's alcoholic with the error term.


# In[ ]:


## comparing results


# In[62]:


# OLS using linearmodels
olsmod = iv.IV2SLS.from_formula(formula='employ ~ 1 + abuse + age + agesq + educ + educsq + married + famsize + white + 
                                northeast + midwest + south + centcity + outercity + qrt1 + qrt2 + qrt3',data=alcohol)
olsres = olsmod.fit(cov_type='unadjusted')

# IV regression using linearmodels
ivr1mod = iv.IV2SLS.from_formula(formula='employ ~ 1 + age + agesq + educ + educsq + married + famsize + white + northeast + 
                                 midwest + south + centcity + outercity + qrt1 + qrt2 + qrt3 + [abuse ~ fathalc]',data=alcohol)
ivr1res = ivr1mod.fit(cov_type='unadjusted')

# IV regression using linearmodels
ivr2mod = iv.IV2SLS.from_formula(formula='employ ~ 1 + age + agesq + educ + educsq + married + famsize + white + northeast + 
                                 midwest + south + centcity + outercity + qrt1 + qrt2 + qrt3 + [abuse ~ fathalc + mothalc]',data=alcohol)
ivr2res = ivr2mod.fit(cov_type='unadjusted')

print(iv.compare({'OLS': olsres, 'IVreg 1 F': ivr1res, 'IVreg 2 F+M': ivr2res}))


# In[ ]:


## Answer 1/5: Endogenous variable "abuse" is highly negatively correlated when both instruments were added and has a relatively 
## low value when only 1 instrument is added. But it has the lowest value when no instruments are included. When one instrument
## is added the coefficient is insignificant for "abuse", but it is significant for other cases.

