## Extrator de características e histograma

import cv2
import numpy as np
import matplotlib.pyplot as plt
import os


class Extractor:

    def conv(self, img, kernel):
        """Operação de convolução entre a imagem e um kernel
        """
        kernel_qtdeLinhas, kernel_qtdeColunas = kernel.shape
        
        img_qtdeLinhas, img_qtdeColunas = img.shape
        
        k1 = round(kernel_qtdeLinhas/2)
        k2 = round(kernel_qtdeColunas/2)
        
        W = np.zeros((img_qtdeLinhas, img_qtdeColunas))
        
        for i in range(k1, img_qtdeLinhas-k1):
            for j in range(k2, img_qtdeColunas-k2):
                soma = 0
                for x in range(kernel_qtdeLinhas):
                    for y in range(kernel_qtdeColunas):
                        soma = soma + kernel[x,y]*img[(i-k1)+x, (j-k2)+y]
                W[i,j] = soma
        return W

    def filtro_sobel(self, img):
        """Obter o resultado a imagem após aplicar o filtro Sobel
        """
        qtdeLinhas, qtdeColunas = img.shape

        kernel_x = np.array([[-1, 0, 1],[-2, 0, 2],[-1, 0, 1]]) #kernel da derivada em relação a x
        kernel_y = np.array([[1, 2, 1],[0, 0, 0], [-1, -2, -1]]) #kernel da derivada em relação a y

        Wx = self.conv(img, kernel_x)
        Wy =self. conv(img, kernel_y)

        Wxy = np.hypot(Wx,Wy)
        Wxy *= 255.0 / np.max(Wxy) #Normalizar
        return Wxy.astype(int)

    def calcular_histograma(self, img):
        """Função para obter o vetor de histograma de uma imagem
        Apenas escala de cinza
        Retorna um vetor de 256 posições com o histograma da imagem
        """
        hist = np.zeros(256)
        qtdeLinhas, qtdeColunas = img.shape
        for i in range(qtdeLinhas):
            for j in range(qtdeColunas):
                hist[img[i,j]] = hist[img[i,j]] + 1
                
        return hist

    def lbp(self, img):
        """Função para obter o Local Binary Pattern (LBP)
        O algoritmo desta função não trata imagens que tenham sofrido rotação ou espelhamento
        """
        qtdeLinhas, qtdeColunas = img.shape
        img2 = np.zeros((qtdeLinhas, qtdeColunas), dtype=int)
        
        for i in range(1, qtdeLinhas-1):
            for j in range(1, qtdeColunas-1):
                A = img[i-1, j]
                B = img[i-1, j+1]
                C = img[i, j+1]
                D = img[i+1, j+1]
                E = img[i+1, j]
                F = img[i+1, j-1]
                G = img[i, j-1]
                H = img[i-1, j-1]
                
                Centro = img[i,j]
                soma = 0
                
                soma += 2**7 if A > Centro else 0
                soma += 2**6 if B > Centro else 0
                soma += 2**5 if C > Centro else 0
                soma += 2**4 if D > Centro else 0
                soma += 2**3 if E > Centro else 0
                soma += 2**2 if F > Centro else 0
                soma += 2**1 if G > Centro else 0
                soma += 2**0 if H > Centro else 0
                
                img2[i,j]  = soma    
                
        return img2

    def extrair_caracteristicas(self, img):
        """Extrai características de uma imagem retornando um histograma
        Os primeiros 256 elementos são o histograma do lbp
        Os últimos 256 elementos são o histograma do Filtro Sobel X e Y
        Retorna um histograma com total de 512 elementos
        """
        img_lbp = self.lbp(img)
        hist_lbp = self.calcular_histograma(img_lbp)
        
        img_sobel = self.filtro_sobel(img)
        hist_sobel = self.calcular_histograma(img_sobel)
        return np.append(hist_lbp, hist_sobel) #Concatenar os dois histogramas


from sklearn.neighbors import KNeighborsClassifier
from sklearn.utils import shuffle
from sklearn.neural_network import MLPClassifier

class CBIR():

    def __init__(self):
        self.extractor = Extractor()
        self.dict = {}
        self.dataset = []
        self.base_hists = []
        self.query = []
        

        self.hists = self.read_hists()

    def distancia(self, a, b):
        M = len(a)
        soma = 0
        for i in range(M):
            soma = soma + ((a[i]-b[i])**2)
        return np.sqrt(soma)

    def normalize(self, hist):
        total = np.sum(hist)
        return np.divide(hist, total)

    def read_hists(self):
        histogramas = []
        
        with open('hists.db') as f:
            for line in f:
                a,b = line.split(',')
                l = a.split()
                l = map(lambda k : float(k), l)
                l = list(l)
                #histogramas[b.strip()] = histogramas.get(b.strip(), l)
                b= b.strip()
                histogramas.append((l, 'corel1000/' + b))
                self.base_hists.append((self.normalize(l), 'corel1000/' + b))

            for hist in histogramas:
                self.dict[hist[1]] = self.dict.get(hist[1], hist[0])

        return histogramas

    #Procurar imagem semelhante na base
    def ranking(self, nome):
        
        img = cv2.imread(nome, 0)
        hist_consulta = self.normalize(self.extractor.extrair_caracteristicas(img))
        self.query = hist_consulta
        d = []

        for i in self.hists:
            hist_query = self.normalize(i[0])
            d.append((self.distancia(hist_consulta, hist_query), i[1], hist_query)) #Calcular distância entre os histogramas

        e = sorted(d)
        return e

    def refilter(self, data):
        useful_data = [x for x in data if x['relevant']] # apenas os marcados como relevantes
        irrelevant = [x for x in data if x['irrelevant']]
        tantofaz = [x for x in data if x not in useful_data and x not in irrelevant]

        for el in useful_data:
            self.dataset.append((tuple(self.normalize(self.dict[el['img']])), 1, el['img']))

        for el in irrelevant:
            self.dataset.append((tuple(self.normalize(self.dict[el['img']])), 0, el['img']))

        self.dataset = list(set(self.dataset))
        print(len(self.dataset))
        #self.dataset = list(set(self.dataset))
        X = list(map(lambda k: k[0], self.dataset))
        Y = list(map(lambda k: k[1], self.dataset))
        knn = KNeighborsClassifier()
        knn.fit(X, Y)

        x_geral = list(map(lambda k: k[0], self.base_hists))
        y_pred = knn.predict(x_geral)

        retorno = []
 
        for i in range(len(y_pred)):
            if y_pred[i] == 1:
                retorno.append((self.distancia(x_geral[i], self.query), self.base_hists[i][1]))

        #print(retorno)

        return list(map(lambda k: k[1], sorted(retorno)))