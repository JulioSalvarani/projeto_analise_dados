import pandas as pd
import math

df = pd.read_csv("base-vendas.csv")

def calcular_entropia (coluna):
    proporcoes = coluna.value_counts(normalize=True)
    entropia = 0
    for p in proporcoes:
        entropia -= p * math.log2(p)
    return entropia

def calcular_gini(coluna):
    proporcoes = coluna.value_counts(normalize=True)
    gini = 1 - sum(p**2 for p in proporcoes)
    return gini

print("Cenário 2 - Entropia - Cidades:", round(calcular_entropia(df['cidade']), 4))
print("Cenário 2 - Gini - Cidades:", round (calcular_gini (df['cidade']), 4))