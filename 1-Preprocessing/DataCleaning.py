import pandas as pd
import numpy as np

def main():
    # Faz a leitura do arquivo
    input_file = '0-Datasets/Mamo.data'
    df = pd.read_csv(input_file, # Nome do arquivo com dados
                     names =['BI-RADS','Age','Shape','Margin','Density','Severity'], # Nome das colunas 
                     usecols = ['Age','Shape','Margin','Density','Severity'], # Define as colunas que serão  utilizadas
                     na_values='?') # Define que ? será considerado valores ausentes
    
    # Imprime as 15 primeiras linhas do arquivo
    print("PRIMEIRAS 15 LINHAS\n")
    print(df.head(15))
    print("\n")

    # Imprime informações sobre dos dados
    print("INFORMAÇÕES GERAIS DOS DADOS\n")
    print(df.info())
    print("\n")

    # Imprime uma analise descritiva sobre dos dados
    print("DESCRIÇÃO DOS DADOS\n")
    print(df.describe())
    print("\n")

    # Imprime a quantidade de valores faltantes por coluna
    print("VALORES FALTANTES\n")
    print(df.isnull().sum())
    print("\n")


    # Tratando valores faltantes da coluna Density
    print("VALORES FALTANTES DA COLUNA Density\n")
    print('Total valores ausentes: ' + str(df['Density'].isnull().sum()))

    method = 'mode' # number or median or mean or mode

    if method == 'number':
        # Substituindo valores ausentes por um número
        df['Density'].fillna(125, inplace=True)

        # Substituindo valores de linhas específicas por um numero
        df.loc[2,'Density'] = 125

    elif method == 'median':
        # Substituindo valores ausentes pela mediana 
        median = df['Density'].median()
        df['Density'].fillna(median, inplace=True)

    elif method == 'mean':
        # Substituindo valores ausentes pela média
        mean = df['Density'].mean()
        df['Density'].fillna(mean, inplace=True)    
      
    elif method == 'mode':
        # Substituindo valores ausentes pela moda
        mode = df['Density'].mode()[0]
        print(mode)
        df['Density'].fillna(mode, inplace=True)    
    
    
    print('Total valores ausentes: ' + str(df['Density'].isnull().sum()))
    print(df.describe())
    print("\n")

    print("\n")
    

if __name__ == "__main__":
    main()
