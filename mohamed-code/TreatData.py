# import pandas and sklearn submodules CountVectorizer and cosine_similarity
import pandas as pd

#load data
def load_data():
    products = pd.read_csv('dataset/uncleaned_Dataset_1k.csv')
    products = clean_data(products)
    return products

# clean the data from empty cells and wrong format and wrong data and duplicates
def clean_data(products):
    
    # delete the first column of the dataframe 
    products = products.drop(columns=["Unnamed: 0"])
    
    # fill all empty cells with empty strings so the vectorizer dont crash later
    products = products.fillna('')
    
    # remove duplication
    products = products.drop_duplicates()
    
    # return the clean Data Frame
    return products
