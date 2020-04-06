from src.load_data import load_raw_csv, load_ventas_byproduct, load_stock_byproduct
import pandas as pd

print("GETTING PRODUCTS TO NOT BE USED")

ventas = load_raw_csv("ventas") 
stock = load_raw_csv("stock")

ventas_byprod = load_ventas_byproduct(ventas)
stock_byprod = load_stock_byproduct(stock)


ventas_byprod = ventas_byprod.loc[:,["producto","fecha_primera_venta","fecha_ultima_venta"]]
stock_byprod = stock_byprod.loc[:,["producto","fecha_primer_stock","fecha_ultimo_stock"]]
prod = ventas_byprod.merge(stock_byprod, how ='outer', on='producto')


print("GETTING PRODUCTS WITH NO STOCK DATA")
nostock = prod.loc[prod.fecha_primer_stock.isna()]
nostock.to_csv("reports/productos/nostock.csv", index=False)
print(nostock)

print("GETTING PRODUCTS NOT SOLD IN MARCH 2020")
nosold = prod.dropna().loc[prod.fecha_ultima_venta < '2020-03-01']
nosold.to_csv("reports/productos/nosold.csv", index=False)
print(nosold)

print("CREATING CONFIG CSV FOR PRODUCTS NOT TO BE STUDIED")
prod = pd.concat([nosold, nostock]).sort_values('producto')
prod[['producto']].to_csv("config/rejected_products.csv", index=False)