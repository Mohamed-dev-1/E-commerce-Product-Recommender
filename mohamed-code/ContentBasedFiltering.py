# import modules
# CountVectorizer Object is used to convert a set of text data into a numerical data a documents with row vectors
from sklearn.feature_extraction.text import CountVectorizer

# the cosine_similiraty will build the similiraty matrix of similiraty scores
from sklearn.metrics.pairwise import cosine_similarity

# we use it to load our dataframe
from TreatData import load_data

# load the clean dataframe
products = load_data()

# we create the vectorizer object with configuration parameters lowercase = True to lower case all the dataset before extracting the words
vectorizer = CountVectorizer(lowercase = True)

# this is the main we will turn text data into scores matrix into phases
# fit() phase : 
# phase 1 : the vectorizer lowercase all the dataset
# phase 2 : the tokenizer extract the words for each document using the token_pattern -> each dpcument has a list of extracted words
# phase 3 : the vectorizer walks throw all the lists of docs and make a set() of tockens to remove duplucation and get the unique words -> features thenhe sort them in an alphanumerical order from a -> z
# phase 4 : he put the features as a keys of a dictionary and their values will be indecies starting from 0 -> then this is vovabulary dics 
# transform() phase :
# now we walk throw all the docs lists of extracted words and create the row vector for each doc by counting the apearance of the vocabulary dics keys in the list of extractred words and we contruct the row vector -> expl [1,0,0,1,1] 
# we finally get a matrix of row vectors each row of the matrix is row vector for doc[i]
category_matrix = vectorizer.fit_transform(products['Category_Path'])

# now we calculate the similiraty scores using the formula A*B / ||A|| ||B|| and we get the similaty scores matrix 
# cosine_simimilarity takes two arguments cosine_similiraty(X, Y) thats why we pass the same matriw twice
# final matrix is scores matrix for each sim(doci , docj) i and j starts from 0 to n number of elements of the category_matrix
cosine_sim = cosine_similarity(category_matrix, category_matrix)

def recommend_content(product_title, df, cosine_sim, top_n=5):
    
    # We search the DataFrame for the row matching the input product_title.
    # .index[0] extracts the integer index of that specific product row.
    target_idx = df[df["Product_Name"] == product_title].index[0]
    
    # Extract similarity vector for the target product
    # cosine_sim is an N x N matrix.
    # Accessing row target_idx gives a 1D array of similarity scores
    product_sim_scores = cosine_sim[target_idx]
    
    # enumerate() transforms the 1D score array into a list of tuples
    # we have a list of tuples fo the the score and the index 
    # [(0, score_0), (1, score_1), ..., (N-1, score_N-1)]
    
    
    enumerated_scores = list(enumerate(product_sim_scores))
    
    # Sort products by similarity score
    # sorted() sorts the list of tuples.
    # key=lambda item: item[1] tells Python to sort by score (the 2nd element).
    # reverse=True sorts in descending order (highest similarity first).
    sorted_scores = sorted(enumerated_scores, key=lambda item: item[1], reverse=True)
    
    # we slice the top_similar_tuples list of tuples into 5 tuples 
    top_similar_tuples = sorted_scores[1 : top_n + 1]
    
    # we put the index of the products in recommended_indices by item[0] -> choose the first element of each tuple -> index
    recommended_indices = [item[0] for item in top_similar_tuples]
    
    # now we make list of recommended_products by waling throw df using recommended_indices
    recommended_products = df.iloc[recommended_indices]

    # return list of recommended_products -> list of df rows reccommended
    return recommended_products

# test

target_product = "Gaming Desktop RTX 4070"

recommendations = recommend_content(target_product, products, cosine_sim, top_n=5)

print(recommendations[["Product_Name", "Category_Path"]])