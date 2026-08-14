# import pandas and sklearn submodules CountVectorizer and cosine_similarity
import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity

#load data
def load_data():
    products = pd.read_csv('uncleaned_Dataset_1k.csv')
    return products