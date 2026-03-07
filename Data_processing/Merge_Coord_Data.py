import pandas as pd

Housing_data = pd.read_csv("./Cleaned_Data_Sets/All_prefectures_buildings_with_migration.csv")
Population_and_coordinate_data = pd.read_csv("../Data/Population_and_coordinate_data/Japanese_Populations_coordinates.csv")

Housing_data['City,Town,Ward,Village code'] = Housing_data['City,Town,Ward,Village code'].astype('Int64')
Population_and_coordinate_data['Area_Codes'] = Population_and_coordinate_data['Area_Codes'].astype('Int64')

merged_df = pd.merge(
    Housing_data,
    Population_and_coordinate_data,
    left_on='City,Town,Ward,Village code',
    right_on='Area_Codes',
    how='inner'
)
merged_df.to_csv("./Cleaned_Data_Deta/All_prefectures_buildings_with_migration_and_coords.csv", index=False)
