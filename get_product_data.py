from src.load_data import load_csv, load_ventas_byproduct



ventas = load_ventas_byproduct()

ventas = load_csv("ventas")
print(ventas.head())

