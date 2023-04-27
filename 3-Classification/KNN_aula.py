from math import sqrt

# função calculo distancia
def distancia_euclideana(vet1, vet2):
    distancia = 0
    for i in range(len(vet1)-1):
        distancia += (vet1[i] - vet2[i])**2
    distancia = sqrt(distancia)
    return distancia

# função retorna k vizinhos mais próximos
def retorna_vizinhos(base_treinamento, amostra_teste, num_vizinhos):
    distancias = list()
    for linha_tre in base_treinamento:
        dist = distancia_euclideana(amostra_teste, linha_tre)
        distancias.append((linha_tre, dist)) # erro
    # ordenação das distancias de forma crescente
    distancias.sort(key=lambda tup: tup[1]) # erro
    # retorna os vizinhos mais proximos
    vizinhos = list()
    for i in range(num_vizinhos):
        vizinhos.append(distancias[i][0]) # erro
    return vizinhos

# função de predição/classificação
def classifica(base_treinamento, amostra_teste, num_vizinhos):
    vizinhos = retorna_vizinhos(base_treinamento, amostra_teste, num_vizinhos)
    rotulos = [v[-1] for v in vizinhos]
    predicao = max(set(rotulos), key=rotulos.count)
    return predicao

dataset = [[2.7, 2.5, 0],[1.4, 2.3, 0],[3.3, 4.4, 0],[1.3, 1.8, 0],[3.0, 3.0, 0],[7.6, 2.7, 1],[5.3, 2.0, 1],[6.9, 1.7, 1],[8.6,-0.2, 1],[7.6, 3.5, 1]]
amostra = [0, 0, 0]
predicao = classifica(dataset, amostra, 3)
print('Resultado da clasificação')
print('Esperado %d\nPredição %d' % (amostra[-1], predicao))
