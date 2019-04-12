
# coding: utf-8

# In[174]:


# Importing libraries
import warnings
warnings.filterwarnings("ignore")
import pandas as pd
import numpy as np
from pgmpy.factors.discrete import TabularCPD
from pgmpy.models import BayesianModel
from pgmpy.sampling import BayesianModelSampling
from pgmpy.estimators import ExhaustiveSearch, K2Score
from pgmpy.inference import VariableElimination
import time


# In[3]:


# Reading the csv files
t2 = pd.read_csv('./cpd/Table2.csv')
t3 = pd.read_csv('./cpd/Table3.csv')
t4 = pd.read_csv('./cpd/Table4.csv')
t5 = pd.read_csv('./cpd/Table5.csv')
t6 = pd.read_csv('./cpd/Table6.csv')
t7 = pd.read_csv('./cpd/Table7.csv')
t8 = pd.read_csv('./cpd/Table8.csv')


# In[4]:


# Dropping the variables columns and saving the numeric values into an array
t2.drop(['values'], axis=1, inplace=True)
t3.drop(['x1'], axis=1, inplace=True)
t4.drop(['x2'], axis=1, inplace=True)
t5.drop(['x3'], axis=1, inplace=True)
t6.drop(['x4'], axis=1, inplace=True)
t7.drop(['x5'], axis=1, inplace=True)
t8.drop(['x6'], axis=1, inplace=True)

# to numpy arrays
t2_array = t2.values
t3_array = t3.values
t4_array = t4.values
t5_array = t5.values
t6_array = t6.values
t7_array = t7.values
t8_array = t8.values


# ## Task 1 - Finding Dependencies

# In[5]:


# Correlation of x1 with other features
t3_corr= np.zeros((t3.shape[0]-1,t3.shape[1]))
corr_x1x2 = 0 
corr_x1x4 = 0
corr_x1x6 = 0
for i in range(0,len(t2_array)):
    for j in range(len(t3_array[0])):
            t3_corr[i][j]= abs(t3_array[i+1][j]-t2_array[i][1])
            
k =i
for i in range(0,len(t2_array)-1):
    for j in range(len(t3_array[0])):
            t3_corr[k+1+i][j]= abs(t3_array[k+i+2][j]-t2_array[i][3])
            
l = i+k+1
for i in range(0,len(t2_array)):
    for j in range(len(t3_array[0])):
            t3_corr[l+1+i][j]= abs(t3_array[i+l+2][j]-t2_array[i][5])

t3_corr1 = np.zeros((t3.shape[0]-1,t3.shape[1]))
for i in range(len(t3_corr)):
    for j in range(len(t3_corr[0])):
        t3_corr1[i][j]= t3_corr[i][j]*t3_array[0][j]
# print(t3_corr1)

for i in range(len(t2_array)):
    for j in range(len(t3_corr[0])):
        corr_x1x2 +=  t3_corr1[i][j]
k =i
for i in range(0,len(t2_array)-1):
    for j in range(len(t3_array[0])):
        corr_x1x4 +=  t3_corr1[k+1+i][j]
            
l = i+k+1
for i in range(0,len(t2_array)):
    for j in range(len(t3_array[0])):
        corr_x1x6 += t3_corr1[l+1+i][j] 

print("correlation between x1 and x2")
print(round(corr_x1x2, 4))
print("correlation between x1 and x4")
print(round(corr_x1x4, 4))
print("correlation between x1 and x6")
print(round(corr_x1x6, 4))


# In[6]:


# Correlation of x2 with other features
t4_corr = np.zeros((t4.shape[0]-1,t4.shape[1]))
corr_x2x3 =0
corr_x2x5 = 0

for i in range(0,len(t2_array)-2):
    for j in range(len(t4_array[0])):
            t4_corr[i][j]= abs(t4_array[i+1][j]-t2_array[i][2])
            
k =i
for i in range(0,len(t2_array)-1):
    for j in range(len(t4_array[0])):
            t4_corr[k+1+i][j]= abs(t4_array[k+i+2][j]-t2_array[i][4])
            
t4_corr1 = np.zeros((t4.shape[0]-1,t4.shape[1]))
for i in range(len(t4_corr)):
    for j in range(len(t4_corr[0])):
        t4_corr1[i][j]= t4_corr[i][j]*t4_array[0][j]

for i in range(len(t2_array)-2):
    for j in range(len(t4_corr[0])):
        corr_x2x3 +=  t4_corr1[i][j]
k =i
for i in range(0,len(t2_array)-1):
    for j in range(len(t3_array[0])):
        corr_x2x5 +=  t4_corr1[k+1+i][j]

print("correlation between x2 and x3")
print(round(corr_x2x3,4))
print("correlation between x2 and x5")
print(round(corr_x2x5,4))


# In[7]:


# Correlation between x3 and other features
t5_corr = np.zeros((t5.shape[0]-1,t5.shape[1]))
cor_x3x2 = 0
cor_x3x5 = 0
cor_x3x6 = 0

for i in range(0,len(t2_array)):
    for j in range(len(t5_array[0])):
            t5_corr[i][j]= abs(t5_array[i+1][j]-t2_array[i][1])
            
k =i
for i in range(0,len(t2_array)-1):
    for j in range(len(t5_array[0])):
            t5_corr[k+1+i][j]= abs(t5_array[k+i+2][j]-t2_array[i][4])
            
l = i+k+1
for i in range(0,len(t2_array)):
    for j in range(len(t5_array[0])):
            t5_corr[l+1+i][j]= abs(t5_array[i+l+2][j]-t2_array[i][5])
            
t5_corr1 = np.zeros((t5.shape[0]-1,t5.shape[1]))
for i in range(len(t5_corr)):
    for j in range(len(t5_corr[0])):
        t5_corr1[i][j]= t5_corr[i][j]*t5_array[0][j]

for i in range(len(t2_array)):
    for j in range(len(t5_corr[0])):
        cor_x3x2 +=  t5_corr1[i][j]
k =i
for i in range(0,len(t2_array)-1):
    for j in range(len(t5_array[0])):
        cor_x3x5 +=  t5_corr1[k+1+i][j]
            
l = i+k+1
for i in range(0,len(t2_array)):
    for j in range(len(t5_array[0])):
        cor_x3x6 += t5_corr1[l+1+i][j] 

print("correlation between x3 and x2")
print(round(cor_x3x2, 4))
print("correlation between x3 and x5")
print(round(cor_x3x5, 4))
print("correlation between x3 and x6")
print(round(cor_x3x6, 4))


# In[8]:


# Correlation between x4 and other features
t6_corr = np.zeros((t6.shape[0]-1,t6.shape[1]))
cor_x4x1 = 0
cor_x4x2 = 0
cor_x4x6 = 0

for i in range(0,len(t2_array)-1):
    for j in range(len(t6_array[0])):
            t6_corr[i][j]= abs(t6_array[i+1][j]-t2_array[i][0])
            
k =i
for i in range(0,len(t2_array)):
    for j in range(len(t6_array[0])):
            t6_corr[k+1+i][j]= abs(t6_array[k+i+2][j]-t2_array[i][1])
            
l = i+k+1
for i in range(0,len(t2_array)):
    for j in range(len(t6_array[0])):
            t6_corr[l+1+i][j]= abs(t6_array[i+l+2][j]-t2_array[i][5])
            
t6_corr1 = np.zeros((t6.shape[0]-1,t6.shape[1]))
for i in range(len(t6_corr)):
    for j in range(len(t6_corr[0])):
        t6_corr1[i][j]= t6_corr[i][j]*t6_array[0][j]

for i in range(len(t2_array)-1):
    for j in range(len(t6_corr[0])):
        cor_x4x1 +=  t6_corr1[i][j]
k =i
for i in range(0,len(t2_array)):
    for j in range(len(t6_array[0])):
        cor_x4x2 +=  t6_corr1[k+1+i][j]
            
l = i+k+1
for i in range(0,len(t2_array)):
    for j in range(len(t6_array[0])):
        cor_x4x6 += t6_corr1[l+1+i][j] 

print("correlation between x4 and x1")
print(round(cor_x4x1, 4))
print("correlation between x4 and x2")
print(round(cor_x4x2, 4))
print("correlation between x4 and x6")
print(round(cor_x4x6, 4))


# In[9]:


# Correlation between x5 and other features
t7_corr = np.zeros((t7.shape[0]-1,t7.shape[1]))
cor_x5_x2 = 0
cor_x5x3 = 0

for i in range(0,len(t2_array)):
    for j in range(len(t7_array[0])):
            t7_corr[i][j]= abs(t7_array[i+1][j]-t2_array[i][1])
            
k =i
for i in range(0,len(t2_array)-2):
    for j in range(len(t7_array[0])):
            t7_corr[k+1+i][j]= abs(t7_array[k+i+2][j]-t2_array[i][2])
            
t7_corr1 = np.zeros((t7.shape[0]-1,t7.shape[1]))
for i in range(len(t7_corr)):
    for j in range(len(t7_corr[0])):
        t7_corr1[i][j]= t7_corr[i][j]*t7_array[0][j]
        
for i in range(len(t2_array)):
    for j in range(len(t7_corr[0])):
        cor_x5_x2 +=  t7_corr1[i][j]
k =i
for i in range(0,len(t2_array)-2):
    for j in range(len(t7_array[0])):
        cor_x5x3 +=  t7_corr1[k+1+i][j]
            
print("correlation between x5 and x2")
print(round(cor_x5_x2, 4))
print("correlation between x5 and x3")
print(round(cor_x5x3, 4))


# In[10]:


# Correlation between x6 and other features
t8_corr = np.zeros((t8.shape[0]-1,t8.shape[1]))
cor_x6x1 = 0
cor_x6x2 = 0
cor_x6x3 = 0
cor_x6x4 = 0

for i in range(0,len(t2_array)-1):
    for j in range(len(t8_array[0])):
            t8_corr[i][j]= abs(t8_array[i+1][j]-t2_array[i][0])
            
k =i
for i in range(0,len(t2_array)):
    for j in range(len(t8_array[0])):
            t8_corr[k+1+i][j]= abs(t8_array[k+i+2][j]-t2_array[i][1])
            
l = i+k+1
for i in range(0,len(t2_array)-2):
    for j in range(len(t8_array[0])):
            t8_corr[l+1+i][j]= abs(t8_array[i+l+2][j]-t2_array[i][2])

m = i+1+l
for i in range(0,len(t2_array)-1):
    for j in range(len(t8_array[0])):
            t8_corr[m+1+i][j]= abs(t8_array[i+m+2][j]-t2_array[i][3])
            
t8_corr1 = np.zeros((t8.shape[0]-1,t8.shape[1]))
for i in range(len(t8_corr)):
    for j in range(len(t8_corr[0])):
        t8_corr1[i][j]= t8_corr[i][j]*t8_array[0][j]

for i in range(len(t2_array)-1):
    for j in range(len(t8_corr[0])):
        cor_x6x1 +=  t8_corr1[i][j]
k =i
for i in range(0,len(t2_array)):
    for j in range(len(t8_array[0])):
        cor_x6x2 +=  t8_corr1[k+1+i][j]
            
l = i+k+1
for i in range(0,len(t2_array)-2):
    for j in range(len(t8_array[0])):
        cor_x6x3 += t8_corr1[l+1+i][j] 

m = i+1+l
for i in range(0,len(t2_array)-1):
    for j in range(len(t8_array[0])):
        cor_x6x4+= t8_corr1[i+m+1][j]
         
print("correlation between x6 and x1")
print(round(cor_x6x1, 4))
print("correlation between x6 and x2")
print(round(cor_x6x2, 4))
print("correlation between x6 and x3")
print(round(cor_x6x3, 4))
print("correlation between x6 and x4")
print(round(cor_x6x4, 4))


# In[12]:


# creating a correlation dataframe to a have reference of all the values
correlations = pd.DataFrame([['x1', 1, 0.1598, 0, 0.1194, 0, 0.1602], 
                             ['x2', 0, 1, 0.2185, 0, 0.0947, 0],
                             ['x3', 0, 0.2346, 1, 0, 0.1128, 0.1192],
                             ['x4', 0.1196, 0.1157, 0, 1, 0, 0.1435],
                             ['x5', 0, 0.8528, 0.1178, 0, 0, 0], 
                             ['x6', 0.1768, 0.1753, 0.139, 0.1431, 0, 0]],
                             columns = ['variable', 'x1', 'x2', 'x3', 'x4', 'x5', 'x6'])
correlations.set_index('variable', inplace=True)
correlations


# ## Task 2 - Building a Bayesian Model

# In[60]:


# Defining CPDs
cpd_x1x2 = TabularCPD(variable = 'x2', variable_card = 5,
                       values = [t3_array[1,0:4],t3_array[2,0:4],t3_array[3,0:4],t3_array[4,0:4],t3_array[5,0:4]],
                       evidence=['x1'], evidence_card=[4])

cpd_x1x4 = TabularCPD('x4', 4,
                      [t3_array[6,0:4],t3_array[7,0:4],t3_array[8,0:4],t3_array[9,0:4]],
                      evidence=['x1'],evidence_card=[4])

cpd_x1x6 = TabularCPD('x6', 5,
                      [t3_array[10,0:4],t3_array[11,0:4],t3_array[12,0:4],t3_array[13,0:4],t3_array[14,0:4]],
                      evidence=['x1'],evidence_card=[4])

cpd_x1 = TabularCPD('x1', 4, [t2_array[0:4, 0]])

cpd_x2x5 = TabularCPD('x5', 4,
                      [t4_array[4,0:5],t4_array[5,0:5],t4_array[6,0:5],t4_array[7,0:5]],
                      evidence=['x2'],evidence_card=[5])

cpd_x2x3 = TabularCPD('x3', 3,
                      [t4_array[1,0:5],t4_array[2,0:5],t4_array[3,0:5]],
                      evidence=['x2'],evidence_card=[5])

cpd_x3 = TabularCPD('x3', 3, [t5_array[0:3, 2]])

cpd_x3x2 = TabularCPD('x2', 3,
                     [t5_array[1,0:3], t5_array[2, 0:3], t5_array[3, 0:3]],
                     evidence=['x3'], evidence_card=[3])

cpd_x3x6 = TabularCPD('x6', 5, 
                     [t5_array[10, 0:3], t5_array[11, 0:3], t5_array[12, 0:3], t5_array[13, 0:3], t5_array[14, 0:3]],
                     ['x3'], [3])

cpd_x3x5 = TabularCPD('x5', 4, 
                     [t5_array[6, 0:3], t5_array[7, 0:3], t5_array[8, 0:3], t5_array[9, 0:3]],
                     ['x3'], [3])

cpd_x6x4 = TabularCPD('x4', 4,
                      [t8_array[13, 0:5], t8_array[14, 0:5], t8_array[15, 0:5], t8_array[16, 0:5]],
                      ['x6'], [5])

cpd_x5x2 = TabularCPD('x2', 5, 
                      [t7_array[1, 0:4], t7_array[2, 0:4], t7_array[3, 0:4], t7_array[4, 0:4], t7_array[5, 0:4]],
                      ['x5'], [4])

cpd_x4x6 = TabularCPD('x6', 5, 
                     [t6_array[10, 0:4], t6_array[11, 0:4], t6_array[12, 0:4], t6_array[13, 0:4], t6_array[14, 0:4]],
                     ['x4'], [4])

cpd_x4x1 = TabularCPD('x1', 4,
                     [t6_array[1, 0:4], t6_array[2, 0:4], t6_array[3, 0:4], t6_array[4, 0:4]],
                     ['x4'], [4])

cpd_x6x1 = TabularCPD('x1', 4,
                     [t8_array[1, 0:5], t8_array[2, 0:5], t8_array[3, 0:5], t8_array[4, 0:5]],
                     ['x6'], [5])

cpd_x6 = TabularCPD('x6', 5, [t8_array[0:5, 0]])

cpd_x2 = TabularCPD('x2', 5, [t4_array[0:5, 0]])

cpd_x6x2 = TabularCPD('x2', 5,
                      [t8_array[5, 0:5], t8_array[6, 0:5], t8_array[7, 0:5], t8_array[8, 0:5], t8_array[9, 0:5]],
                      ['x6'], [5])

# Normalizing the CPDs
cpd_x1x2.normalize(True)
cpd_x1x4.normalize(True)
cpd_x1x6.normalize(True)
cpd_x1.normalize(True)
cpd_x2x5.normalize(True)
cpd_x5x2.normalize(True)
cpd_x2x3.normalize(True)
cpd_x3.normalize(True)
cpd_x3x2.normalize(True)
cpd_x3x6.normalize(True)
cpd_x6x4.normalize(True)
cpd_x4x6.normalize(True)
cpd_x4x1.normalize(True)
cpd_x6x1.normalize(True)
cpd_x6.normalize(True)
cpd_x2.normalize(True)
cpd_x6x2.normalize(True)
cpd_x3x5.normalize(True)


# ##### Creating Models and generating data

# In[31]:


# First Model
model1 = BayesianModel()
model1.add_nodes_from(['x1','x2','x3','x4','x5','x6'])
model1.add_edges_from([('x1','x2'),('x1','x4'),('x1','x6'),('x2','x3'),('x2','x5')])
model1.add_cpds(cpd_x1,cpd_x1x2,cpd_x1x4,cpd_x1x6,cpd_x2x3,cpd_x2x5)
inference = BayesianModelSampling(model1)
# print(inference.forward_sample(size=1000, return_type='dataframe'))
data1 = inference.forward_sample(size=1000,return_type='dataframe')

# Second Model
model2 = BayesianModel()
model2.add_nodes_from(['x1','x2','x3','x4','x5','x6'])
model2.add_edges_from([('x1','x2'),('x1','x4'),('x6','x1'),('x2','x3'),('x2','x5')])
model2.add_cpds(cpd_x6,cpd_x1x2,cpd_x1x4,cpd_x6x1,cpd_x2x3,cpd_x2x5)
inference = BayesianModelSampling(model2)
# print(inference.forward_sample(size=1000, return_type='dataframe'))
data2 = inference.forward_sample(size=1000,return_type='dataframe')

# Third Model
model3 = BayesianModel()
model3.add_nodes_from(['x1','x2','x3','x4','x5','x6'])
model3.add_edges_from([('x2', 'x3'),('x2','x5'),('x3','x6'),('x6','x4'),('x4','x1')])
model3.add_cpds(cpd_x2, cpd_x2x3, cpd_x2x5, cpd_x3x6, cpd_x6x4, cpd_x4x1)
inference = BayesianModelSampling(model3)
# print(inference.forward_sample(size=1000, return_type='dataframe'))
data3 = inference.forward_sample(size=1000,return_type='dataframe')

# Fourth Model
model4 = BayesianModel()
model4.add_nodes_from(['x1','x2','x3','x4','x5','x6'])
model4.add_edges_from([('x3', 'x2'),('x2','x5'),('x3','x6'),('x6','x4'),('x4','x1')])
model4.add_cpds(cpd_x3, cpd_x3x2, cpd_x2x5, cpd_x3x6, cpd_x6x4, cpd_x4x1)
inference = BayesianModelSampling(model4)
# print(inference.forward_sample(size=1000, return_type='dataframe'))
data4 = inference.forward_sample(size=1000,return_type='dataframe')

# Fifth Model
model5 = BayesianModel()
model5.add_nodes_from(['x1','x2','x3','x4','x5','x6'])
model5.add_edges_from([('x1','x2'),('x1','x6'),('x6','x4'),('x2','x3'),('x3','x5')])
model5.add_cpds(cpd_x1,cpd_x1x2,cpd_x1x6,cpd_x6x4,cpd_x2x3,cpd_x3x5)
inference = BayesianModelSampling(model5)
# print(inference.forward_sample(size=1000, return_type='dataframe'))
data5 = inference.forward_sample(size=1000,return_type='dataframe')


# ##### Evaluating the models using K2 score on the generated data

# In[70]:


# Evaluating the models on the data sets generated by them
data = pd.concat([data1, data2, data3, data4, data5])
data.shape

k2 = K2Score(data)

print('Model 1 K2 Score: ' + str(k2.score(model1))) # model 1 is the best model
print('Model 2 K2 Score: ' + str(k2.score(model2)))
print('Model 3 K2 Score: ' + str(k2.score(model3)))
print('Model 4 K2 Score: ' + str(k2.score(model4)))
print('Model 5 K2 Score: ' + str(k2.score(model5)))


# ##### Find the high and low probability patterns of 'th'

# In[153]:


# Finding 'th' highest frequency pattern
frequency = data.groupby(['x1', 'x2', 'x3', 'x4', 'x5', 'x6']).size().to_frame('count').reset_index()
print('The high probability has a pattern of 011031 and occurs 83 times. Please see below.')
print(frequency.sort_values('count', ascending = False).iloc[0:2, ])


# Find 'th' lowest frequency pattern
print('\nThe low probability has a pattern of 042133 and occurs 1 time. Please see below.')
print(frequency.sort_values('count', ascending = True).iloc[0:2, ])


# ### Additonal:- Finding the best Bayesian Model for the obtained 'data'

# In[121]:


# Finding the best Bayesian model which describes the data using hillclimbsearch
hc = HillClimbSearch(data, scoring_method=K2Score(data))
best_model = hc.estimate()

print(best_model.edges())
# the best edges obtained by the search are different from any of the model edges we defined above


# In[165]:


# Bayesian Model and parameter estimation
model = BayesianModel([('x1', 'x6'), ('x1', 'x2'), ('x2', 'x3'), ('x2', 'x5'), ('x4', 'x1')])

# Bayesian Parameter Estimation
model.fit(data, estimator=BayesianEstimator, prior_type='K2', equivalent_sample_size=50)

# check_model checks for the network structure and CPDs and verifies that the CPDs are correctly 
# defined and sum to 1.
print('Model check: ', model.check_model(), '\n')

# Evaluating the fit
k2 = K2Score(data)

print('Model K2 Score: ', str(k2.score(model)), '\n') #the K2 score is least by a small margin when compared the above models

print('The cpds obtained using Bayesian Parameter estimation are: \n')
cpds = model.get_cpds()
for cpd in cpds:
    print(cpd)


# In[173]:


from pgmpy.inference import VariableElimination
infer = VariableElimination(model)
print(infer.query(['x2'])['x2'])

print('\n', infer.query(['x5'], evidence={'x2': 1, 'x3': 1, 'x1':0, 'x4':0, 'x6':1}) ['x5']) #this is the pattern for highest
#'th' probability. As expected the probabilty of x5 taking the value of 3 is highest because it occurs 83 times when the other
# variable values are fixed and the next highest being x5=0 occuring 81 times.


# In[239]:


# Computational time of inference using Bayesian Network
import time
time_start = time.clock()
infer = VariableElimination(model)
infer.query(['x3'])['x3']
infer.query(['x2'], evidence={'x1':0, 'x6':1}) ['x2']
time_elapsed = (time.clock() - time_start)
print('Computation time for inference using "Bayesian Network": ', time_elapsed)


# # Task 3 - Converting to a Markov Network

# In[172]:


mm = model.to_markov_model()
mm.nodes()
mm.edges() 

# Finding inferences using Markov Network
infer = VariableElimination(mm)
print(infer.query(['x2'])['x2'])
print('\n', infer.query(['x5'], evidence={'x2': 1, 'x3': 1, 'x1':0, 'x4':0, 'x6':1}) ['x5'])


# In[240]:


# Computational time of inference using Markov Network
import time
time_start = time.clock()
infer = VariableElimination(mm)
infer.query(['x3'])['x3']
infer.query(['x2'], evidence={'x1':0, 'x6':1}) ['x2']
time_elapsed = (time.clock() - time_start)
print('Computation time for inference using "Markov Network": ', time_elapsed)


# # Task 4

# In[138]:


from pgmpy.estimators import HillClimbSearch
from pgmpy.estimators import ExhaustiveSearch
from pgmpy.estimators import K2Score
from pgmpy.models import BayesianModel
from pgmpy.estimators import BayesianEstimator


# In[139]:


# Loading AND_Features dataset and finding the best structure for the model
df = pd.read_csv('AND-Features.csv')
df = df.iloc[:, 2:11]

hc = HillClimbSearch(df, scoring_method=K2Score(df))
best_model = hc.estimate()

print(best_model.edges())


# In[163]:


# Bayesian Model and parameter estimation
model1 = BayesianModel([('f3', 'f4'), ('f3', 'f9'), ('f3', 'f8'), ('f5', 'f9'), ('f5', 'f3'), 
                       ('f9', 'f8'), ('f9', 'f7'), ('f9', 'f1'), ('f9', 'f6'), ('f9', 'f2'), ('f9', 'f4')])

# Bayesian Parameter Estimation
est = BayesianEstimator(model1, df)

cpd_f1 = est.estimate_cpd('f1', prior_type='K2', equivalent_sample_size=50)
cpd_f2 = est.estimate_cpd('f2', prior_type='K2', equivalent_sample_size=50)
cpd_f3 = est.estimate_cpd('f3', prior_type='K2', equivalent_sample_size=50)
cpd_f4 = est.estimate_cpd('f4', prior_type='K2', equivalent_sample_size=50)
cpd_f5 = est.estimate_cpd('f5', prior_type='K2', equivalent_sample_size=50)
cpd_f6 = est.estimate_cpd('f6', prior_type='K2', equivalent_sample_size=50)
cpd_f7 = est.estimate_cpd('f7', prior_type='K2', equivalent_sample_size=50)
cpd_f8 = est.estimate_cpd('f8', prior_type='K2', equivalent_sample_size=50)
cpd_f9 = est.estimate_cpd('f9', prior_type='K2', equivalent_sample_size=50)

# Associating the CPDs with the network
model1.add_cpds(cpd_f1, cpd_f2, cpd_f3, cpd_f4, cpd_f5, cpd_f6, cpd_f7, cpd_f8, cpd_f9)

# check_model checks for the network structure and CPDs and verifies that the CPDs are correctly 
# defined and sum to 1.
print('Model 1 check: ', model1.check_model())

#######################
##### Model 2 #####
#######################
model2 = BayesianModel([('f3', 'f4'), ('f3', 'f9'), ('f3', 'f8'), ('f5', 'f3'), 
                       ('f9', 'f7'), ('f9', 'f1'), ('f9', 'f6'), ('f9', 'f2'), ('f9', 'f4')])

# Bayesian Parameter Estimation using fit()
model2.fit(df, estimator=BayesianEstimator, prior_type='K2', equivalent_sample_size=50)

# check_model checks for the network structure and CPDs and verifies that the CPDs are correctly 
# defined and sum to 1.
print('Model 2 check: ', model2.check_model())

#######################
##### Model 3 #####
#######################
model3 = BayesianModel([('f3', 'f4'), ('f3', 'f9'), ('f3', 'f8'), ('f5', 'f9'), ('f3', 'f5'), 
                       ('f9', 'f8'), ('f9', 'f7'), ('f9', 'f1'), ('f9', 'f6'), ('f9', 'f2'), ('f9', 'f4')])

# Bayesian Parameter Estimation
model3.fit(df, estimator=BayesianEstimator, prior_type='K2', equivalent_sample_size=50)

# check_model checks for the network structure and CPDs and verifies that the CPDs are correctly 
# defined and sum to 1.
print('Model 3 check: ', model3.check_model())


# In[164]:


# Evaluating the fit
k2 = K2Score(df)

print('Model 1 K2 Score: ' + str(k2.score(model1)))
print('Model 2 K2 Score: ' + str(k2.score(model2)))
print('Model 3 K2 Score: ' + str(k2.score(model3)))


# In[142]:


from pgmpy.inference import VariableElimination
infer = VariableElimination(model1)
print(infer.query(['f5'])['f5'])


# In[143]:


print(infer.query(['f5'], evidence={'f4': 0, 'f1': 1}) ['f5'])

