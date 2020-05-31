import pandas as pd

# File processing online downloaded data to a format ML model can accept ot for training.

mql = pd.read_csv('marketing_qualified_leads.csv', usecols=['mql_id', 'landing_page_id', 'origin'])
closed_deal = pd.read_csv('olist_closed_deals.csv', usecols=['mql_id'])

# Adding a label column: closed_deal as 1 else 0 for prediction model
closed_deal['label'] = 1

# merging to make features
merged = pd.merge(mql, closed_deal, how='left', on='mql_id')
merged.fillna({'label':0}, inplace=True)

merged.dropna(inplace=True)

pos_data = merged[merged['label'].eq(1)]
_count_pos = len(pos_data)
neg_data = merged[merged['label'].eq(0)].sample(n=_count_pos, random_state=1234)

final_data = pd.concat([pos_data, neg_data])
final_data = final_data.sample(frac=1)

# Saving to get picked up by model
final_data.to_csv('test_merge.csv')
