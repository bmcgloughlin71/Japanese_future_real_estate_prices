import pandas as pd
df_migration = pd.read_csv("../Data/Internal_Migration_Data_2008_2024/Number_of_Migants_to_Muncipalties_per_year.csv")

df_housing_prices = pd.read_csv("./Cleaned_Data_Sets/All_prefectures_buildings.csv")

# First, rename columns to align for merging
df_migration_renamed = df_migration.rename(columns={
    'area_code': 'City,Town,Ward,Village code',
    'value': 'Migration',
    'Year': 'MigrationYear'
})

# Create a 'TargetYear' in housing_prices for joining
df_housing_prices['TargetYear'] = df_housing_prices['Year'] - 1 #assumption is that migration has a 1 year lag on housing prices

# Attempt the direct merge on area_code and target year
merged = pd.merge(
    df_housing_prices,
    df_migration_renamed,
    how='left',
    left_on=['City,Town,Ward,Village code', 'TargetYear'],
    right_on=['City,Town,Ward,Village code', 'MigrationYear']
)

# Now fill missing Migration values using the earliest year per area_code
# First, get the earliest year migration value per area_code
earliest_migration = (
    df_migration_renamed.sort_values('MigrationYear')
    .drop_duplicates('City,Town,Ward,Village code')
    [['City,Town,Ward,Village code', 'Migration']]
)

# Fill in missing Migration values in merged dataframe
merged = pd.merge(
    merged,
    earliest_migration,
    how='left',
    on='City,Town,Ward,Village code',
    suffixes=('', '_earliest')
)

# Use the found migration value or fall back to earliest
merged['Migration'] = merged['Migration'].combine_first(merged['Migration_earliest'])

# Drop helper columns
merged = merged.drop(columns=['Migration_earliest', 'TargetYear', 'MigrationYear'])

merged.to_csv("./Cleaned_Data_Sets/All_prefectures_buildings_with_migration.csv", index=False)
