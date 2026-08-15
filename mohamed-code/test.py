from TreatData import load_data, clean_data

products = load_data()
products = clean_data(products)
print(products.info())

# test if the data is clean or need more treatment


