import pandas as pd

data = pd.read_csv('../Cleaned_Data_Sets/All_prefectures_buildings.csv')
data = data.sample(frac=1, random_state=5).reset_index(drop=True)
out_prefix = "All_prefecture_Housing"

# Do a Training, dev, test split of 80/10/10
train_size = int(0.8 * len(data))
dev_size = int(0.1 * len(data))

# Split the data
train_data = data[:train_size]
dev_data = data[train_size:train_size + dev_size]
test_data = data[train_size + dev_size:]

#Output
train_data.to_csv(f'{out_prefix}_training_data.csv', index=False)
dev_data.to_csv(f'{out_prefix}_dev_data.csv', index=False)
test_data.to_csv(f'{out_prefix}_test_data.csv', index=False)

print(f'Train size: {len(train_data)}')
print(f'Dev size: {len(dev_data)}')
print(f'Test size: {len(test_data)}')
