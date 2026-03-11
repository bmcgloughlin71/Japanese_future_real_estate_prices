import pandas as pd

def assign_nearest_population(row):
    year = row['Year']  # Transaction year
    pop_years = [2005, 2010, 2015, 2020]
    pop_cols = ['Population_2005', 'Population_2010', 'Population_2015', 'Population_2020']

    # Compute absolute differences
    diffs = [abs(year - y) for y in pop_years]

    sorted_indices = sorted(range(len(diffs)), key=lambda i: diffs[i])

    for idx in sorted_indices:
        pop_value = row[pop_cols[idx]]
        if pd.notna(pop_value) and pop_value != 0:
            return pop_value

    raise ValueError(
        f"No valid population found for Area_Code {row['City,Town,Ward,Village code']} "
        f"at transaction year {year}. Check population dataset."
    )

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

merged_df['Population'] = merged_df(assign_nearest_population, axis=1)
merged_df.to_csv("./Cleaned_Data_Sets/All_prefectures_buildings_with_migration_and_coords.csv", index=False)
