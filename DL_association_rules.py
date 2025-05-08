# Importing libraries
import pandas as pd
from mlxtend.frequent_patterns import fpgrowth
from mlxtend.frequent_patterns import association_rules

# Loading the refined dataset
data = pd.read_excel("/Users/denysenko/Desktop/refined_database.xlsx")

# Checking the data
print("Data loaded successfully")
print(data.head())

# Preprocess
binary_data = data.notnull()
binary_data = binary_data.astype(int)

# Just to be sure
print("Binary data preview:")
print(binary_data.head())

# Apply FP-Growth algorithm
# min_support is set to 0.1
frequent_itemsets = fpgrowth(binary_data, min_support=0.1, use_colnames=True)

# View some frequent itemsets
print("Some frequent itemsets found:")
print(frequent_itemsets.head())

# Generate association rules
rules = association_rules(frequent_itemsets, metric="lift", min_threshold=1.0)

# Drop missing values just in case
rules = rules.dropna()

# For better readability: Convert frozensets to strings
def set_to_string(x):
    return ', '.join(list(x))

rules['antecedents'] = rules['antecedents'].apply(set_to_string)
rules['consequents'] = rules['consequents'].apply(set_to_string)

# Show top rules by lift
print("Top Association Rules:")
top_rules = rules.sort_values(by="lift", ascending=False)
top_rules = top_rules.head(10)

for i, row in top_rules.iterrows():
    print("If a paper has:", row['antecedents'], "then it often also has:", row['consequents'], "(Lift:", round(row['lift'], 2), ")")
