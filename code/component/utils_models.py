import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

def Results_Table(lst):
    results_table = pd.concat(lst)
    return results_table