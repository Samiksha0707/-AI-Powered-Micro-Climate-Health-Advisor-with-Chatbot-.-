from backend.data_loader import load_data

data = load_data("data/Health_climate_data.csv")

print(data.head())
