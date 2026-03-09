import pandas as pd
df_House_with_coords = pd.read_csv("./Cleaned_Data_Sets/All_prefectures_buildings_with_migration_and_coords.csv")

designated_cities =  pd.read_csv("../Data/designated_cities_and_tokyo.txt", delim_whitespace=True, comment='#',
                        names=["CityName", "Latitude", "Longitude"])

print("Calculating distance to nearest designated city . . .\n")
df_House_with_coords[['Distance_to_designated_city', 'Nearest_designated_city']] = df_House_with_coords.progress_apply(
    lambda row: find_closest_city_and_distance(row['latitude'], row['longitude']), axis=1)

print("Handling special cases...\n")
greater_tokyo_set = {'Yokohama', 'Kawasaki', 'Saitama', 'Chiba', 'Sagamihara'}

df_House_with_coords['Close_to_Tokyo'] = (df_House_with_coords['Nearest_designated_city'] == 'Tokyo').astype(int)
df_House_with_coords['Close_to_greater_Tokyo_area'] = df_House_with_coords['Nearest_designated_city'].isin(greater_tokyo_set).astype(int)
df_House_with_coords['Close_to_designated_city_flag'] = (df_House_with_coords['Distance_to_designated_city'] < 5).astype(int)

cols_to_exclude = [
    "Population_2020", "Population_2015", "Population_2010", "Population_2005", "Area_Codes", "Destination"
]

df_House_with_coords.drop(columns=cols_to_exclude, inplace=True)
df_House_with_coords.to_csv("./Cleaned_Data_Sets/Final_Cleaned_Data_Set.csv")
