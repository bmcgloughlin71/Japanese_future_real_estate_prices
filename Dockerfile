FROM python:3.10-slim

WORKDIR /app

COPY requirements.txt /app/requirements.txt
RUN pip install --no-cache-dir -r /app/requirements.txt

COPY app /app/app
COPY artifacts /app/artifacts
COPY Regression_Analysis/Model_and_Weights/Japanese_Housing_Price_Model.keras \
    /app/Regression_Analysis/Model_and_Weights/

COPY Data/Population_and_coordinate_data/Japanese_Populations_coordinates.csv \
    /app/Data/Population_and_coordinate_data/
COPY Data/Population_and_coordinate_data/designated_cities_and_tokyo.txt \
    /app/Data/Population_and_coordinate_data/
COPY Data/Internal_Migration_Data_2008_2024/Number_of_Migants_to_Muncipalties_per_year.csv \
    /app/Data/Internal_Migration_Data_2008_2024/

ENV PORT=8000
EXPOSE 8000

CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]
