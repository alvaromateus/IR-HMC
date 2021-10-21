# -*- coding: utf-8 -*-
import time
import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn import preprocessing
from sklearn.preprocessing import PolynomialFeatures
import array as arr
#from scipy import stats
import math
from datetime import datetime
from sklearn import metrics
import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)

class Parametros(object):
    # Parametrizações iniciais
    metodo = 0 # 0-Regressão Linar   |   1-Regressão Polinomial  |  2-Regressão linear múltipla
    grau_polinomio = 1 # usado somente para regressão polinomial
    coeficiente_correlacao = 0.3

    nome_base = "Cellcycle"
    df = pd.read_csv("datasets/GOCellcycle(atributosFaltando).txt")
    df_hierarquia = pd.read_csv("datasets/hierarquia_cellcycle.txt")
    arquivo_divisao = "datasets/GOCellcycleInstanciasTreinamento.txt"

    #nome_base = "Church"
    #df = pd.read_csv("datasets/GOChurch(atributosFaltando).txt")
    #df_hierarquia = pd.read_csv("datasets/hierarquia_church.txt")
    #arquivo_divisao = "datasets/GOChurchInstanciasTreinamento.txt"

    #nome_base = "Eisen"
    #df = pd.read_csv("datasets/GOEisen(atributosFaltando).txt")
    #df_hierarquia = pd.read_csv("datasets/hierarquia_eisen.txt")
    #arquivo_divisao = "datasets/GOEisenInstanciasTreinamento.txt"

    #nome_base = "Expr"
    #df = pd.read_csv("datasets/GOExpr(atributosFaltando).txt")
    #df_hierarquia = pd.read_csv("datasets/hierarquia_expr.txt")
    #arquivo_divisao = "datasets/GOExprInstanciasTreinamento.txt"

    #nome_base = "Gasch1"
    #df = pd.read_csv("datasets/GOGasch1(atributosFaltando).txt")
    #df_hierarquia = pd.read_csv("datasets/hierarquia_gash1.txt")
    #arquivo_divisao = "datasets/GOGasch1InstanciasTreinamento.txt"

    #nome_base = "Gasch2"
    #df = pd.read_csv("datasets/GOGasch2(atributosFaltando).txt")
    #df_hierarquia = pd.read_csv("datasets/hierarquia_gash2.txt")
    #arquivo_divisao = "datasets/GOGasch2InstanciasTreinamento.txt"

    #nome_base = "Seq"
    #df = pd.read_csv("datasets/GOSeq(atributosFaltando).txt")
    #df_hierarquia = pd.read_csv("datasets_GO/seq_GO/hierarquia_seq.txt")
    #arquivo_divisao = "datasets/GOSeqInstanciasTreinamento.txt"

    #nome_base = "Spo"
    #df = pd.read_csv("datasets/GOSpo(atributosFaltando).txt")
    #df_hierarquia = pd.read_csv("datasets/hierarquia_spo.txt")
    #arquivo_divisao = "datasets/GOSpoInstanciasTreinamento.txt"
class Correlacao(object): 
    def __init__(self, nome, indice): 
        self.nome = nome 
        self.indice = indice 

    def __repr__(self): 
        return "nome:%s indice:%s" % (self.nome, self.indice)

    def get_nome(self):   
        return self.nome

    def get_indice(self): 
        return self.indice

def retorna_ascendentes(conjunto, classe):  # retorna classe pai
    pais = []  # variável que será retornada do tipo array pois será possível haver mais de um ascendente
    conjunto = conjunto[conjunto['item'].str.contains(classe)]  # filtra somente os registros com a classe solicitada

    for index, row in conjunto.iterrows():  # Percorre todos os registros
        pais.append(
            row[0].split("/")[0])  # caso encontre adiciona a variável que será retornada. Utiliza o divisor "/"
    return pais

def verifica_rotulos_ascendentes(base, rotulo, base_hierarquia):  # Verificaçãose há outros exemplos com rótulo ascendnete(Verificação hierárquica)
    rotulos = []  # variável que será retornada
    # Caso o rótulo ascendente não possuir exemplos será acionado o próprio método recursivamente
    if (base[base['class'].str.contains(rotulo)].empty):
        return (verifica_rotulos_ascendentes(base, retorna_ascendentes(base_hierarquia, rotulo)[0],base_hierarquia))
    else:  # Caso o  rótulo ascendente possuir exemplos este é adicionado a variável de retorno
        return (base[base['class'].str.contains(rotulo)])

def polyfit(x, y, degree):
    results = {}
    try:
        coeffs = np.polyfit(x, y, degree)
    except ValueError:
        results['determination'] = 0.0
        #print("\nErro!\n")
        return results
    else:    
        # Polynomial Coefficients
        results['polynomial'] = coeffs.tolist()

        # r-squared
        p = np.poly1d(coeffs)
        # fit values, and mean
        yhat = p(x)                      # or [p(z) for z in x]
        ybar = np.sum(y)/len(y)          # or sum(y)/len(y)
        ssreg = np.sum((yhat-ybar)**2)   # or sum([ (yihat - ybar)**2 for yihat in yhat])
        sstot = np.sum((y - ybar)**2)    # or sum([ (yi - ybar)**2 for yi in y])
        try:
            results['determination'] = ssreg / sstot
        except ValueError:
            results['determination'] = 0.0
            #print("\nErro!\n")
            return results

        return results

def isNaN(num):
    return num != num

def verificacao_hierarquica_multirrotulo(base, rotulos_dado_faltante, base_hierarquia):
    conjunto = []
    classes_encontradas = []
    classes_nao_encontradas = []

    # Verificação Multirrótulo
    for i in rotulos_dado_faltante:
        conjuntos_semelhantes = base[base.iloc[:, base.columns.get_loc('class')].str.contains(i)]
        if not (conjuntos_semelhantes.empty):
            conjunto.append(conjuntos_semelhantes)
            classes_encontradas.append(i)  # adiciona o item a variável de control
        else:
            classes_nao_encontradas.append(i)
    #print(len(conjunto))

    if (len(conjunto) > 1):  # une dataframes
        conjunto_ = pd.merge_ordered(conjunto[0], conjunto[1], fill_method="ffill")
        contador = 2
        while len(conjunto) > contador:
            conjunto_ = pd.merge_ordered(conjunto_, conjunto[contador])
            contador = contador+1
        conjunto = conjunto_
    else:
        conjunto = conjunto[0]

    # Verificação Hierárquica
    dados_rotulos = []
    lista_final = list(set(classes_nao_encontradas) - set(classes_encontradas))  # retorna a diferença das duas listas
    if ((len(lista_final)) != 0):
        if (conjunto[conjunto.iloc[:, conjunto.columns.get_loc('class')].str.contains(lista_final[0])].empty):
            ascendentes = retorna_ascendentes(base_hierarquia, lista_final[0])
            for j in ascendentes:  # verificação caso seja estrutura do tipo DAG (vários ascendentes / Pais)
                dados_rotulos.append(verifica_rotulos_ascendentes(base, j, base_hierarquia))  # adiciona os rótulos a variável de retorno
            if len(dados_rotulos)>1:
                conjunto_ = pd.merge_ordered(dados_rotulos[0], dados_rotulos[1], fill_method="ffill")
                contador = 2
                conjunto_ = conjunto_.drop_duplicates()
                if len(dados_rotulos)>2:
                    while len(dados_rotulos) > contador:
                        conjunto_ = pd.merge_ordered(conjunto_, dados_rotulos[contador], fill_method="ffill")
                        conjunto_ = conjunto_.drop_duplicates()
                        contador = contador + 1
                else:
                    conjunto = pd.merge_ordered(conjunto, conjunto_, fill_method="ffill")
            else:
                conjunto = pd.merge_ordered(conjunto, dados_rotulos[0], fill_method="ffill")
    return conjunto.fillna(conjunto.mean(0)) # caso o conjunto resultante possua dados faltantes faz imputação pela média        

def histogram_intersection(a, b):
    v = np.minimum(a, b).sum().round(decimals=1)
    return v

def correlacao(conjunto, indice_atributo_a, indice_atributo_b, metodo, grau):
    x = conjunto.fillna(0).iloc[:, indice_atributo_a].values
    y = conjunto.fillna(0).iloc[:, indice_atributo_b].values    

    if (metodo == 0 or metodo == 2):        
        x = x.reshape(-1, 1)
        modelo = LinearRegression()
        modelo.fit(x, y)
        indice = modelo.score(x, y)
        if indice == 1: # condição para não retornar o próprio atributo
            return 0
        else:
            return indice
    if (metodo == 1): # usa método específico
        #indice = polyfit(x,y,grau)['determination'].round(5)
        from sklearn.metrics import r2_score
        indice = r2_score(x, y)
        
        if indice == 1:
            return 0
        else:
            return indice

def melhor_correlacao_v2(indice_atributo, conjunto, metodo, grau, exemplo_imputado):
    contador = 0
    lista = []
    lista_final = []

    # laço para criar lista com todas correlações
    while (contador < conjunto.iloc[0, :].size - 1):        
        nome_coluna = conjunto.columns[contador]        
        corr = correlacao(conjunto, indice_atributo, contador, metodo, grau)            
        lista.append(Correlacao(nome_coluna, corr))
        contador = contador + 1                
    lista = sorted(lista, key=lambda x: x.indice, reverse=True) 
    for i in lista:
        if i.indice > Parametros.coeficiente_correlacao: # Se houver correlação boa
            lista_final.append(i)
    if len(lista_final) > 0: 
        return lista_final
    else: # se não encontrar correlação boa retorna primeira posição
        lista_final.append(lista[0])
        return lista_final        
       
def melhor_correlacao(indice_atributo, conjunto, metodo, grau, exemplo_imputado):               
    correlation = conjunto.corr(method="pearson")
    quantidade_colunas = correlation.shape[0]        
    selecao = correlation[correlation.iloc[indice_atributo]==1].values
    lista = []
    
    if metodo == 0 or metodo == 1: # regressão linear ou polinomial        
        try:
            indice_correlacao = np.sort(selecao[0])[quantidade_colunas-2]
            nome_coluna = correlation.columns.values[correlation.iloc[indice_atributo]==indice_correlacao]
            correlacao = Correlacao(nome_coluna,indice_correlacao*indice_correlacao)
            lista.append(correlacao)
            return lista
        except:
            correlacao = Correlacao('e',0)
            lista.append(correlacao)
            return lista
    if metodo == 2: # regressão múltipla
        if len(conjunto) > 3:
            try:
                indice = quantidade_colunas-2                        
                while indice > 0:
                    if np.sort(selecao[0])[indice] >= Parametros.coeficiente_correlacao: # Se houver correlação boa                        
                        selecao = correlation[correlation.iloc[indice_atributo]==1].values
                        indice_correlacao = np.sort(selecao[0])[indice]
                        nome_coluna = correlation.columns.values[correlation.iloc[indice_atributo]==indice_correlacao]        
                        
                        correlacao = Correlacao(nome_coluna,indice_correlacao*indice_correlacao)
                        lista.append(correlacao)                    
                    else:                         
                        break
                    indice -= 1
                if len(lista)>0:
                    return lista
                else:
                    correlacao = Correlacao('indeterminado',0)
                    lista.append(correlacao)
                    return lista
            except: 
                correlacao = Correlacao('e',0)
                lista.append(correlacao)
                return lista
        else:
            correlacao = Correlacao('c',0)
            lista.append(correlacao)
            return lista

def regressao_polinomial(coluna_melhor_correlacao, indice_atributo, conjunto, variavel_independente, grau):
    coluna_melhor_correlacao = conjunto.columns.get_loc(coluna_melhor_correlacao)

    x = conjunto.fillna(0).iloc[:, indice_atributo].values
    y = conjunto.fillna(0).iloc[:, coluna_melhor_correlacao].values

    poly_reg = PolynomialFeatures(degree=grau)

    x = x.reshape(-1, 1)
    try:
        x_poly = poly_reg.fit_transform(x)
        poly_reg.fit(x_poly,y)    

        lin_reg_2 = LinearRegression()
        
        lin_reg_2.fit(x_poly,y)
        resultado = lin_reg_2.predict(poly_reg.fit_transform([[variavel_independente]]))[0]        
        return resultado
    except ValueError:
        resultado = conjunto.iloc[:, coluna_melhor_correlacao].mean()
        return resultado

def regressao_linear(coluna_melhor_correlacao, indice_atributo, conjunto, variavel_independente):
    coluna_melhor_correlacao = conjunto.columns.get_loc(coluna_melhor_correlacao)

    x = conjunto.fillna(0).iloc[:, indice_atributo].values
    y = conjunto.fillna(0).iloc[:, coluna_melhor_correlacao].values

    x = x.reshape(-1, 1)
    modelo = LinearRegression()
    modelo.fit(x, y)

    return modelo.predict([[variavel_independente]])[0]

def regressao_multipla(lista_correlacao, indice_atributo, conjunto):
    x = pd.DataFrame()
    variaveis_independentes = []
    # define as variáveis independentes (dados da instância com atributo faltante nas colunas em que há boa correlação)
    for i in lista_correlacao:        
        x[i.get_nome()] = conjunto[i.get_nome()].fillna(conjunto.mean())        
        variaveis_independentes.append(conjunto.columns.get_loc(i.get_nome()))
    x = x.values
    try:
        #coluna_melhor_correlacao = conjunto.columns.get_loc(lista_correlacao)
        y = conjunto.fillna(0).iloc[:, indice_atributo].values
        
        modelo = LinearRegression()
        modelo.fit(x, y)
        
        variaveis_independentes = np.array([*variaveis_independentes])
        #variaveis_independentes = variaveis_independentes.reshape(1, -1)
            
        return modelo.predict([[*variaveis_independentes]])[0]
    except: # se der erro faz a média
        resultado = conjunto.iloc[:, lista_correlacao[0].get_nome()].mean()
        print("Erro na regressão múltipla")
        return resultado

def normalize(df_input):
    header = df_input.columns
    coluna_classe = pd.DataFrame(df_input['class'])
    base_drop = pd.DataFrame(df_input.drop(columns=['class'])) # Retira coluna da classe para fazer normalização    
    
    min_max_scaler = preprocessing.MinMaxScaler()   
    x_scaled = np.around(min_max_scaler.fit_transform(base_drop), decimals=3) # Normaliza com Min/Max    
    #x_scaled = (base_drop - base_drop.min()) / (base_drop.max() - base_drop.min())    
    #x_scaled = base_drop/base_drop.max()
    #x_scaled = base_drop / base_drop.max(axis=0)

    df_input = pd.DataFrame(x_scaled)
    df_input = df_input.join(coluna_classe, lsuffix='_caller', rsuffix='_other') # Adiciona coluna classe novamente
    df_input.columns = header

    return df_input

def any_nan(conjunto):
    nan = 0
    not_nan = 0
    for x in conjunto:
        if isNaN(x):
            nan = nan+1
        else:
            not_nan = not_nan+1    
    
    if not_nan>0:
        return False
    else:
        return True           

def undo_split(arquivo_instancias_treinamento, base_imputada):
    instancias_treinamento = pd.read_csv(arquivo_instancias_treinamento, header=None)

    treinamento = base_imputada.loc[base_imputada.index[instancias_treinamento[0]-1]]
    teste = base_imputada.loc[base_imputada.index.difference(instancias_treinamento[0]-1)]
    return teste,treinamento

# Instancia e zera contadores
coluna = 0
quantidade_regressao = 0
quantidade_media = 0
quantidade_moda = 0

parametros = Parametros() # classe com parâmetros e bases de dados iniciais

while (coluna < parametros.df.iloc[0, :].size-1):
    linha = 0
    while(linha < parametros.df['class'].size):    
        if isNaN(parametros.df.loc[linha][coluna]): # verifica se a linha e coluna possui dado faltante        
            rotulos_dado_faltante = parametros.df.loc[linha][parametros.df.columns.get_loc('class')].split('@')  # seleciona linha x coluna
            exemplo_imputado = parametros.df.loc[linha]  # armazena o exemplo a ser imputado 
            df_ = parametros.df.drop(linha)  # cria copia removendo exemplo com dado faltante do conjunto
            
            #define subconjunto
            conjunto = verificacao_hierarquica_multirrotulo(df_, rotulos_dado_faltante, parametros.df_hierarquia)  # define conjunto semelhante            
            print("Linha: "+str(linha)+" | Coluna: "+str(coluna) + " | "+parametros.df.columns.values[coluna] +" | Time: " +datetime.now().strftime("%d/%m/%Y - %H:%M:%S"))            

            # define atributos categórios das bases de dados 
            atributos_categoricos = (['spo_failed_pcr','spo_blast_homology_within_genome','spo_overlaps_another_orf','failed_pcr','blast_homology_within_genome','church_chip_affymetrix_chip','strand'])
            
            # verifica possível dado categórico no conjunto, pois caso houver a matriz de correlação possuirá dados nulos                        
            #verificacao_correlacao = conjunto[conjunto.iloc[:,coluna]==0].count()[0] + conjunto[conjunto.iloc[:,coluna]==1].count()[0]
            
            if not (parametros.df.columns.values[coluna] in atributos_categoricos):
                # define atributo com melhor correlação                
                #atributo_melhor_correlacao = melhor_correlacao(coluna, conjunto, parametros.metodo, parametros.grau_polinomio, exemplo_imputado)
                atributo_melhor_correlacao = melhor_correlacao_v2(coluna, conjunto, parametros.metodo, parametros.grau_polinomio, exemplo_imputado)
                
                # Se correlação for boa (maior ou igual que 0.5) usa regressão
                if (atributo_melhor_correlacao[0].get_indice() >= Parametros.coeficiente_correlacao):                     

                    # define a variável independente (dado da instância com atributo faltante na coluna que possui melhor correlação)
                    variavel_independente = parametros.df.fillna(parametros.df.mean())[atributo_melhor_correlacao[0].get_nome()].loc[linha]                    

                    # regressão linear (se método igual a 0 ou houver somente um atributo com correlação superior a 0.5)
                    if (parametros.metodo == 0 or len(atributo_melhor_correlacao) == 1 and parametros.metodo != 1): 
                        regressao = regressao_linear(atributo_melhor_correlacao[0].get_nome(), coluna, conjunto.drop(columns=['class']), variavel_independente)

                        # faz a imputação do valor utilizando o modelo de regressão linear                        
                        parametros.df.iloc[linha, coluna] = round(regressao,4)
                        print("Usando REGRESSÃO LINEAR. Melhor coeficiente de correlação: "+ str(atributo_melhor_correlacao[0].get_indice()) + " com atributo "+atributo_melhor_correlacao[0].get_nome() + " - valor imputado: "+str(regressao))
                    # regressão polinomial
                    if (parametros.metodo == 1): 
                        regressao = regressao_polinomial(atributo_melhor_correlacao[0].get_nome(), coluna, conjunto.drop(columns=['class']), variavel_independente, parametros.grau_polinomio)
                                                
                        # faz a imputação do valor utilizando o modelo de regressão polinomial
                        parametros.df.iloc[linha, coluna] = round(regressao,4)
                        print("Usando REGRESSÃO POLINOMIAL. Melhor coeficiente de correlação: "+ str(atributo_melhor_correlacao[0].get_indice()) + " com atributo "+atributo_melhor_correlacao[0].get_nome() + " - valor imputado: "+str(regressao))
                    # regressão múltipla (se método igual a dois e houver mais de um atributo com correlação superior a 0.5)                    
                    if (parametros.metodo == 2 and len(atributo_melhor_correlacao) > 1): 
                        regressao = regressao_multipla(atributo_melhor_correlacao, coluna, conjunto.drop(columns=['class']))
                                                
                        # faz a imputação do valor pelo utilizando o modelo de regressão múltipla
                        parametros.df.iloc[linha, coluna] = round(regressao,4)
                        print("Usando REGRESSÃO MÚLTIPLA. Melhor coeficiente de correlação: "+ str(atributo_melhor_correlacao[0].get_indice()) + " com atributo "+atributo_melhor_correlacao[0].get_nome() + " - valor imputado: "+str(regressao))
                    quantidade_regressao = quantidade_regressao+1

                # se correlação não for boa (<0.5) usa média 
                else:
                    media = conjunto.iloc[:, coluna].mean(skipna=True)                    
                    if not (isNaN(media)): # Caso seja possível definir a média
                        # faz a imputação do valor utilizando a média do conjunto
                        parametros.df.iloc[linha, coluna] = media 
                    else:
                        # se não for possível determinar a média insere o valor 0
                        media = 0
                        parametros.df.iloc[linha, coluna] = 0
                    print("Usando MÉDIA. Melhor coeficiente de correlação: "+str(atributo_melhor_correlacao[0].get_indice())  + " com atributo "+atributo_melhor_correlacao[0].get_nome()+ " - valor imputado: "+str(media)) 
                    quantidade_media = quantidade_media+1
            # Se for atributo categórica faz imputação pela moda
            else:                
                moda = conjunto.iloc[:, coluna].mode()
                if not (moda.empty): # caso seja possível definir a moda                    
                    # faz a imputação do valor utilizando a moda
                    parametros.df.iloc[linha, coluna] = moda[0]
                else:
                    moda = 1                    
                    parametros.df.iloc[linha, coluna] = 1
                print("Usando MODA. Atributo categórico encontrado - valor imputado: "+str(moda))
                quantidade_moda = quantidade_moda+1            
        linha = linha + 1        
    coluna = coluna + 1
    print("Coluna concluída")

# imprime base de dados imputada
print(parametros.df)

# normaliza os dados
df = normalize(parametros.df)

# divide em base e treinamento (conforme arquivo que define a divisão)
teste, treinamento = undo_split(parametros.arquivo_divisao, df) # divide em treinamento e teste

# salva os resultados em dois arquivos .csv (treinamento e teste) inserindo informações do método utilizado ao final do nome do arquivo
metodo = ""
if parametros.metodo == 0:
    metodo = "-linear"
if parametros.metodo == 1:
    metodo = "-polinomial"
if parametros.metodo == 2:
    metodo = "-multipla"
treinamento.to_csv(r'treinamento-'+parametros.nome_base+"-M"+str(quantidade_media)+"-R"+str(quantidade_regressao)+"-MODA"+str(quantidade_moda)+(metodo+str(parametros.grau_polinomio))+'.csv', index=False, header=True) # salva nos arquivos correspondenttes
teste.to_csv(r'teste-'+parametros.nome_base+"-M"+str(quantidade_media)+"-R"+str(quantidade_regressao)+"-MODA"+str(quantidade_moda)+(metodo+str(parametros.grau_polinomio))+'.csv', index=False, header=True)