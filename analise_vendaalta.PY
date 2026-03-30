''' 
Objetivo da Análise: 
Procurar fatores associados a vendas de alto valor e 
características aumentam a probabilidade de uma venda ser de alto valor.
'''
############
# DATABASE #
############

# [ Passo 1 — Importar as bibliotecas necessárias ] #

import pandas as pd

# [ Passo 2 — Carregar o dataset ] #

df = pd.read_csv("vendas.csv")
#print(df.head())

# [ Passo 3 — Crie as variáveis valor _total e venda_alta ] #

df["valor_total"] = (df["quantidade"] *df["preco_unitario"]).astype(int)
df["venda_alta"] = (df["valor_total"] > 2000).astype(int)
#print(df.head())

# [ Passo 4 — Verificar a distribuição da variável venda_alta ] #

#print(df["venda_alta"].value_counts())

'''
Síntese da parte:
Apartir desses passos, já podemos definir o "Objetivo" em código, por assim dizer, 
o motivo pelo qual a árvore de decisão foi criada.
'''
############
# ENTROPIA #
############


# [ Passo 1 — Importar biblioteca matemática ] #

import numpy as np

# [ Passo 2 — Calcular as proporções das classes ] #

proporcoes = df["venda_alta"].value_counts(normalize=True)
print(proporcoes)

# [ Passo 3 — Criar função para calcular entropia ] #

def calcular_entropia(coluna):
    proporcoes = coluna.value_counts(normalize=True)
    entropia = -np.sum(proporcoes * np.log2(proporcoes))
    return entropia

# [ Passo 4 — Calcular a entropia do dataset como um todo ] #

entropia_inicial = calcular_entropia(df["venda_alta"])
print("Entropia inicial:", entropia_inicial)


"""
Síntese da parte:
Através do calculo da entropia inicial (0.65), percebemos que a base está relativamente organizada.
Mas ainda mais importante, agora podemos calcular o gain de cada categoria.
"""

############
#   GAIN   #
############

# [ Passo 1 — Calcular entropia ponderada/entropia por categoria,  [ EntropiaPai - EntropiaFilho ], também usar função para automaticamente retornar o ganho ] #

def calcular_ganho_informacao(df, coluna_teste, coluna_alvo, entropia_ref):
    entropia_ponderada = 0
    proporcoes = df[coluna_teste].value_counts(normalize=True)
    
    for valor, peso in proporcoes.items():
        subset = df[df[coluna_teste] == valor]
        entropia_grupo = calcular_entropia(subset[coluna_alvo])
        entropia_ponderada += peso * entropia_grupo
        
    return entropia_ref - entropia_ponderada


# [ Passo 2 — Checar o ganho de informação por coluna ] #

ganho_cat = calcular_ganho_informacao(df, "categoria", "venda_alta", entropia_inicial)
ganho_pag = calcular_ganho_informacao(df, "forma_pagamento", "venda_alta", entropia_inicial)
ganho_cid = calcular_ganho_informacao(df, "cidade", "venda_alta", entropia_inicial)

print(f"Ganho Categoria: {ganho_cat}")
print(f"Ganho Pagamento: {ganho_pag}")
print(f"Ganho Cidade: {ganho_cid}")

'''
Síntese da parte: 
Apartir dos resultados

Ganho Categoria: 0.10917033867559889
Ganho Pagamento: 0.011176819492026646
Ganho Cidade: 0.04183297077682668

Podemos concluir " categoria " é a melhor candidata para o primeiro nó.
'''

############
# SUBSETS  #
############

# [ Passo 1 - Criar subconjuntos/nós para definir entre as categorias qual teria o maior GAIN (CATEGORIA -> *) ] #

df_eletronicos = df[df["categoria"] == "Eletrônicos"]
df_moveis = df[df["categoria"] == "Móveis"]

    #Checagem rápida se os Dataframes estão sendo definidos corretamente

print(f"O Data frame de eletrônicos está assim:{df_eletronicos}")
print(f"O Data frame de móveis está assim:{df_moveis}")

# [ Passo 2 - Fazer o próximo "nó" da árvore ] #

    #Para Eletrônicos

entropia_eletronicos = calcular_entropia(df_eletronicos["venda_alta"])

ganho_pag_e = calcular_ganho_informacao(df_eletronicos, "forma_pagamento", "venda_alta", entropia_eletronicos)
ganho_prod_e = calcular_ganho_informacao(df_eletronicos, "produto", "venda_alta", entropia_eletronicos)
ganho_cid_e = calcular_ganho_informacao(df_eletronicos, "cidade", "venda_alta", entropia_eletronicos)

print("Eletrônicos:")
print(f"Pagamento: {ganho_pag_e}")
print(f"Produto: {ganho_prod_e}")
print(f"Cidade: {ganho_cid_e}")

    #Para Móveis

entropia_moveis = calcular_entropia(df_moveis["venda_alta"])

ganho_pag_m = calcular_ganho_informacao(df_moveis, "forma_pagamento", "venda_alta", entropia_moveis)
ganho_prod_m = calcular_ganho_informacao(df_moveis, "produto", "venda_alta", entropia_moveis)
ganho_cid_m = calcular_ganho_informacao(df_moveis, "cidade", "venda_alta", entropia_moveis)

print("Móveis:")
print(f"Pagamento: {ganho_pag_m}")
print(f"Produto: {ganho_prod_m}")
print(f"Cidade: {ganho_cid_m}")

# Mas por que Móveis deu 0 de entropia? Veja bem

print("\nContagem de classes em Móveis:")
print(df_moveis['venda_alta'].value_counts())

# como podemos ver, móveis não tem NENHUM item com "venda_alta", o que indica que o ganho de informação e sua entropia são 0.

# Por fim, Eletrônicos por Produto foi o melhor DISPARADO, ENTÃO SEGUIMOS NA LÓGICA:

# [ Passo 3 - Definir um nó folha Puro (CATEGORIA {Eletrônicos} -> PRODUTO -> *) ] #

def analisar_no(df_subset, nome):
    print(f"\n{'='*30}")
    print(f" NÓ: {nome}")
    print(f"{'='*30}")
    
    # 1. Mostrar o volume de dados e a distribuição
    total_amostras = len(df_subset)
    distribuicao = df_subset["venda_alta"].value_counts().to_dict()
    
    # Garante que mostre 0 e 1 mesmo que um deles não exista no grupo
    vendas_baixas = distribuicao.get(0, 0)
    vendas_altas = distribuicao.get(1, 0)
    
    print(f"Total de registros: {total_amostras}")
    print(f"Distribuição: [Baixas: {vendas_baixas} | Altas: {vendas_altas}]")
    
    entropia = calcular_entropia(df_subset["venda_alta"])
    print(f"Entropia atual: {entropia:.4f}")
    
    # 2. Regra de parada com explicação humana
    if entropia == 0:
        resultado = "VENDA ALTA (1)" if vendas_altas > 0 else "VENDA BAIXA (0)"
        print(f"RESULTADO: Este é um nó folha puro!")
        print(f"   Decisão: {resultado}")
        return
    
    # 3. Análise de ganhos
    ganhos = {}
    print("\nAnalisando possíveis divisões:")
    for col in ["forma_pagamento", "cidade"]:
        ganho = calcular_ganho_informacao(df_subset, col, "venda_alta", entropia)
        ganhos[col] = ganho
        print(f" - Ganho com '{col}': {ganho:.4f}")
    
    melhor = max(ganhos, key=ganhos.get)
    
    if ganhos[melhor] <= 0:
        # Se não há ganho, decidimos pela maioria
        maioria = 1 if vendas_altas > vendas_baixas else 0
        print(f"Sem ganho de informação. Folha por maioria: {maioria}")
    else:
        print(f"! MELHOR DIVISÃO: {melhor}")

# [ Passo final - Demonstrar qual produto é o nó folha ideal ] #

for produto, subset in df_eletronicos.groupby("produto"):
    analisar_no(subset, f"Produto: {produto}")

'''
Síntese final:
Através de uma análise completa do banco de dados, pode-se dizer que o nó folha ideal para " venda_alta " É " Notebook " !!
O que indica que: o item que é constantemente mais associado a vendas de alto valor é o Notebook. 
'''