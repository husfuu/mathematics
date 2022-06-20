import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from utility import *

history_model1 = pd.read_excel('./data/model-results/history_model1.xlsx') 
history_model2 = pd.read_excel('./data/model-results/history_model2.xlsx') 
history_model3 = pd.read_excel('./data/model-results/history_model3.xlsx')
history_model4 = pd.read_excel('./data/model-results/history_model4.xlsx')

# add epochs to all history_model
history_model1 = addEpochs(history_model1)
history_model2 = addEpochs(history_model2)
history_model3 = addEpochs(history_model3)
history_model4 = addEpochs(history_model4)

# plot all results from model that we've built
plotResults(history_model1)
plotResults(history_model2)
plotResults(history_model3)
plotResults(history_model4)