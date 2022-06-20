import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from utility import *

history_Adam = pd.read_excel('./data/model-results/resAdam_1.xlsx') 
history_Sgd = pd.read_excel('./data/model-results/resSgd_1.xlsx')

# clean the history model data
cleanData(history_Adam)
cleanData(history_Sgd)

# plot all results from model that we've built
plotResults(history_Adam)
plotResults(history_Sgd)