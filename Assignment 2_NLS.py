#!/usr/bin/env python
# coding: utf-8

# In[2]:


import numpy  as np
import pandas as pd

import statsmodels.formula.api as smf
import scipy.stats as stats

import linearmodels.iv as iv

import wooldridge as woo


# In[56]:


df= pd.read_csv('./nls80.csv')


# In[57]:


#summary statistics
print(df.info())
print(df.describe())


# In[58]:


###Removing black observations from the dataset
nlsb = df[df['black']==0].copy()


# In[59]:


#Removing mother's education observations from the new dataset
nlsm = nlsb.dropna(subset=['meduc'])


# In[66]:


#Summary Statistics
print(nlsm.info())
print(nlsm.describe())


# In[60]:


## correlation between iq,kww, educ
iq = nlsm ['iq'].values
kww = nlsm ['kww'].values
educ = nlsm ['educ'].values


# In[63]:


cor1 = np.corrcoef(iq, kww)
print(cor1)


# In[64]:


cor2 = np.corrcoef(kww, educ)
print(cor2)


# In[65]:


cor3 = np.corrcoef(educ, iq)
print(cor3)


# In[ ]:


## Answer 2/1: Number of observations 758 and variables 16. Average of all variables listed on the table.
## Correlation between IQ and Knowledge of World is 0.34, IQ and Education is 0.51 and Knowledge of world and education is 0.37.


# In[67]:


## OLS model without IQ variable
nlsmmod = smf.ols(formula='lwage ~ educ + exper + tenure + south + urban', data=nlsm)
nlsmres = nlsmmod.fit(cov_type='HC1')
print(nlsmres.summary())


# In[68]:


## OLS model with IQ variable
nlsmmod = smf.ols(formula='lwage ~ educ + exper + tenure + south + urban + iq', data=nlsm)
nlsmres = nlsmmod.fit(cov_type='HC1')
print(nlsmres.summary())


# In[ ]:


## Answer 2/2: R square and adjusted R square is around 0.2 in both OLS models. 20% of the variation is explained by the model. 
## High F-statistic value with p value of F statistic is almost 0 in both OLS models.
## Almost no change in constant term even after adding IQ variable, though it's not statistically significant.
## Almost all coefficients are statistically significant at all level of signficance.
## IQ variable has almost no affect on the wage rate as it has coefficients of almost 0 which is significant also.


# In[81]:


#IV regression with 2sls method
#First stage equation 

nlsmfsmod = smf.ols(formula = 'iq ~ educ + exper + tenure + south+ urban + meduc + kww + age', data = nlsm)
nlsmfsres = nlsmfsmod.fit()
nlsmfstab = pd.DataFrame({'b'   : round(nlsmfsres.params,4),
                      'se'  : round(nlsmfsres.bse, 4),
                      't'   : round(nlsmfsres.tvalues, 2),
                      'pval': round(nlsmfsres.pvalues,4)})
print()
print('First stage regression')
print(nlsmfstab)


# In[78]:


nlsm['iqres'] = nlsmfsres.resid
nlsm['iqhat'] = nlsmfsres.fittedvalues


# In[84]:


#Second stage equation (2SLS version 2)
nlsmts2mod = smf.ols(formula='lwage ~ iq + educ + exper + tenure + south+ urban + iqhat',data=nlsm)
nlsmts2res = nlsmts2mod.fit()
nlsmts2tab = pd.DataFrame({'b'   : round(nlsmts2res.params,4),
                      'se'  : round(nlsmts2res.bse, 4),
                      't'   : round(nlsmts2res.tvalues, 2),
                      'pval': round(nlsmts2res.pvalues,4)})
print()
print('Second stage regression - Version 2')
print(nlsmts2tab)


# In[83]:


# instrument validity      
# Sargan's J-stat test


# In[85]:


nlsm['tslsrhat'] = nlsmts2res.resid
sjmod = smf.ols(formula='tslsrhat ~ iq + educ + exper + tenure + south+ urban + iqhat + meduc + kww + age',data=nlsm)
sjres = sjmod.fit()

sj_lm = sjres.nobs * sjres.rsquared
sj_pv = 1 - stats.chi2.cdf(x=sj_lm,df=1)

print()
print('Aux reg Rsqrd: {}'.format(round(sjres.rsquared,6)))
print('Sargan J-test: {}'.format(round(sj_lm,4)))
print('      p-value: {}'.format(round(sj_pv,4)))


# In[ ]:


## Answer 2/3-1: Instruments are valid as the p value of J test is almost 0.00


# In[117]:


# test for endogeneity
endhyp = ['iqhat=0']
endtest = nlsmts2res.f_test(endhyp2)
wistat = witest.statistic
wipval = witest.pvalue

print()
print('Wooldridge-Hausman-Wu test for endogeneity')
print('F-stat : {}'.format(endtest.statistic))
print('p-value: {}'.format(endtest.pvalue))


# In[98]:


## IV regression using GMM
#    this assumes a robust covariance matrix
#
gmrmod = iv.IVGMM.from_formula(formula='lwage ~ 1 + educ + exper + tenure + south+ urban + [iq ~ meduc + kww + age]',weight_type='robust',data=nlsm)
gmrres = gmrmod.fit()

gmrrestab = pd.DataFrame({'b'   : round(gmrres.params,5),
                          'se'  : round(gmrres.std_errors, 5),
                          't'   : round(gmrres.tstats, 2),
                          'pval': round(gmrres.pvalues,4)})
print(gmrrestab)


# In[99]:


print()
print('Testing for overidentifying restrictions')
print(gmrres.j_stat)
#print(gmrres.j_stat.stat)
#print(gmrres.j_stat.pval)
#print(gmrres.c_stat(['iq']))


# In[ ]:


## Answer 2/3-3: The overidentifying restrictions test has p value of almost 0.00 with F statistic of 14.17.
## There is enough evidence to reject the null hypothesis saying the additional variables are endogenous.


# In[102]:


#IV regression with 2sls method when educ is endogenous variable
#First stage equation 

nlsmfsmod2 = smf.ols(formula = 'educ ~ iq + exper + tenure + south+ urban + meduc + kww + age', data = nlsm)
nlsmfsres2 = nlsmfsmod2.fit()
nlsmfstab2 = pd.DataFrame({'b'   : round(nlsmfsres2.params,4),
                      'se'  : round(nlsmfsres2.bse, 4),
                      't'   : round(nlsmfsres2.tvalues, 2),
                      'pval': round(nlsmfsres2.pvalues,4)})
print()
print('First stage regression')
print(nlsmfstab2)


# In[113]:


nlsm['edures'] = nlsmfsres2.resid
nlsm['eduhat'] = nlsmfsres2.fittedvalues


# In[104]:


#Second stage equation (2SLS version 2)
nlsmts2mod2 = smf.ols(formula='lwage ~ iq + age + exper + tenure + south+ urban + eduhat',data=nlsm)
nlsmts2res2 = nlsmts2mod2.fit()
nlsmts2tab2 = pd.DataFrame({'b'   : round(nlsmts2res2.params,4),
                      'se'  : round(nlsmts2res2.bse, 4),
                      't'   : round(nlsmts2res2.tvalues, 2),
                      'pval': round(nlsmts2res2.pvalues,4)})
print()
print('Second stage regression - Version 2')
print(nlsmts2tab2)


# In[106]:


# instrument validity      
# Sargan's J-stat test
nlsm['tslsrhat2'] = nlsmts2res2.resid
sjmod2 = smf.ols(formula='tslsrhat2 ~ iq + educ + exper + tenure + south + urban + iqhat + meduc + kww + age',data=nlsm)
sjres2 = sjmod2.fit()

sj_lm = sjres2.nobs * sjres2.rsquared
sj_pv = 1 - stats.chi2.cdf(x=sj_lm,df=1)

print()
print('Aux reg Rsqrd: {}'.format(round(sjres2.rsquared,6)))
print('Sargan J-test: {}'.format(round(sj_lm,4)))
print('      p-value: {}'.format(round(sj_pv,4)))


# In[ ]:


## Answer 2/4-1: Instruments are valid as the p value of J test is almost 0.00


# In[112]:


# test for endogeneity
endhyp2 = ['eduhat=0']
endtest2 = nlsmts2res2.f_test(endhyp2)
wistat2 = witest2.statistic
wipval2 = witest2.pvalue

print()
print('Wooldridge-Hausman-Wu test for endogeneity')
print('F-stat : {}'.format(endtest2.statistic))
print('p-value: {}'.format(endtest2.pvalue))


# In[109]:


## IV regression using GMM
#    this assumes a robust covariance matrix
#
gmrmod2 = iv.IVGMM.from_formula(formula='lwage ~ 1 + age + exper + tenure + south+ urban + [educ ~ meduc + kww + iq]',weight_type='robust',data=nlsm)
gmrres2 = gmrmod2.fit()

gmrrestab2 = pd.DataFrame({'b'   : round(gmrres2.params,5),
                          'se'  : round(gmrres2.std_errors, 5),
                          't'   : round(gmrres2.tstats, 2),
                          'pval': round(gmrres2.pvalues,4)})
print(gmrrestab2)


# In[114]:


print()
print('Testing for overidentifying restrictions')
print(gmrres2.j_stat)
#print(gmrres.j_stat.stat)
#print(gmrres.j_stat.pval)
#print(gmrres.c_stat(['educ']))


# In[ ]:


## Answer 2/4-3: The overidentifying restrictions test has p value of almost 0.4507 with F statistic of 1.59.
## There is not enough evidence to reject the null hypothesis saying the additional variables are exogenous.

