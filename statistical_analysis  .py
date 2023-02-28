#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jan  8 16:36:52 2021

@author: ibtihalkhan
"""

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import os
os.getcwd()
os.chdir("/Users/ibtihalkhan/Downloads")
df= pd.read_csv("QA_DATASET.csv")
df.head()
df.info()
'''
The data was collected containing question answer pair and question type
526 observations
7 variables
ArticleTitle -Title of the article– ordinal categorical 
Question - WH type question - ordinal categorical
Answer - answer of the given question – ordinal categorical 
question_keyword - store WH keyword– ordinal categorical
rule_label- queuestion type –ordinal cat
manually_label – question type labbeled manually – ordinal cat
Correctly_labelled - correctly labelled question – nominal cat
'''
#univariate plot

plt.hist(df.Correctly_labelled)
plt.xlabel('corecrt labelled')
plt.ylabel('count')
plt.title('no of question correctly labelled')
#most of the question are correctly labelled with the constructed rule based model

'''
Response Variable is correctly_labelled, whether the question label correctly or not by the rule-based model 

REsearch question how questions_keyword influences by correctly_labelled (i.e. cheking the feasibility of rule-based model)
'''

sns.countplot(x='question_keyword',hue='Correctly_labelled',data=df)
#ALL WH question are almost correctly labelled

ad= pd.crosstab(index=df['Correctly_labelled'], columns=df["question_keyword"])
ad
ad= pd.crosstab(index=df['Correctly_labelled'], columns=df["question_keyword"],
                        normalize='columns',margins=True)
ad
#look like there is a relationship between question_keyword and Correctly_labelled,
#as ALL WH question are almost correctly labelled with the constructed rule based model
#giving high accuracy of precise answer 



#perform logistic regression

from statsmodels.api import Logit
from statsmodels.formula.api import logit


'''
HO: Null model is prefferd 
H1: model with rule_label and mannuly_label is preferred   
'''

logit_model=logit('Correctly_labelled~ C(rule_label)+C(manually_label)',df).fit()
logit_model.summary2()


#LRT comparing to null model
#As p-value=1.3254e-61>0.05,we reject Null Hypothesis i.e model with rule_label and manually_label is preferred

#pi here is probability of correctly label
#logit(pi)=log(success odd of correctly label)=
#0.8504-2.6721rule_labelObjective+5.2687manually_labelObjective
np.exp(0.8504)
#est  success odds for Correctly label is
#2.3405828978634684 when we take rule label and manually label as objective
np.exp(-2.6721)
#est  success odds for correctly label change by a factor
#0.06910694823051845 when we compare a Descritve to a Objective type for
#rule_label 
np.exp(5.2687)
#est  success odds for correctly label change by a factor
#194.16338590453654 when we compare a Descritve to a Objective type for
#manual_label 

'''
All terms are significantly different 0 (i.e. they have a relationship
with Correctly labelled variable
'''


#perform one-way ANOVA test

import statsmodels.api as sm

import scipy.stats as stats
from statsmodels.formula.api import ols

#perform one-way ANOVA 
onewaymodel = ols('Correctly_labelled ~ C(question_keyword)', data=df).fit()
#fitted values
model_fitted_vals = onewaymodel.fittedvalues
#model residuals
model_residuals = onewaymodel.resid
#standardised residuals
model_norm_residuals = onewaymodel.get_influence().resid_studentized_internal


sns.regplot(x=model_fitted_vals,y=model_residuals,
                ci=False,lowess=True,
                line_kws={'color': 'red', 'lw': 1, 'alpha': 0.8})
plt.xlabel("Fitted Values")
plt.ylabel("Residuals")
#spread of residuals not looks the same so nothappy with equal variance
#assumption

stats.probplot(model_norm_residuals, plot=sns.mpl.pyplot)
plt.show()
#evidence of strong deviations from normaility as 
#not follows the line not happy with this
#assumptin

onewaymodel.summary()
#Looking at the summary we have line equation is
#y=0.7656-0.0671question_keywordwhat+0.2140question_keywordwhen+0.2344question_keywordwhere+0.2344 question_keywordwhich +0.2344 question_keywordwho+0.2344 question_keywordwhy
#The model explain only 12.5% variance


