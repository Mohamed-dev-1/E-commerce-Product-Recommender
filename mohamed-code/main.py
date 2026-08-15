from ContentBasedFiltering import recommend_content, cosine_sim
from TreatData import load_data

products = load_data()

target_product = "Gaming Desktop RTX 4070"

recommendations = recommend_content(target_product, products, cosine_sim, top_n=5)

print(recommendations[["Product_Name", "Category_Path"]])

