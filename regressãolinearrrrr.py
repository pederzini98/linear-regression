from re import L
from turtle import color
from numpy import double
import pandas as pd
import matplotlib.pyplot as plt

data = pd.read_csv("kc_house_data.csv")

def loss_function(m, b, pontos):
    erro_total = 0
    for i in range(len(pontos)):
        x = pontos.iloc[i].sqft_living
        y = pontos.iloc[i].price
        erro_total += (y - (m *x + b)) ** 2
    return erro_total / float(len(pontos))

def gradiente_descendente(m_atual, b_atual, pontos, coeficiente_aprendizado):
    m_gradiente = 0
    b_gradiente = 0
    n = len(pontos)
    for i in range(n):
        x = pontos.iloc[i].sqft_living
        y = pontos.iloc[i].price

        m_gradiente += double(- (2/n) * x * (y - (m_atual * x + b_atual)))
        b_gradiente += double(- (2/n) * (y - (m_atual * x + b_atual)))
    m = double(m_atual - m_gradiente * coeficiente_aprendizado)
    b = double( b_atual - b_gradiente * coeficiente_aprendizado)
    return m,b

m = 0
b = 0
L = 0.001
iteracoes = 300

for i in range(iteracoes):
    if i % 50 == 0:
        print(i)
    m, b =  gradiente_descendente(m, b, data, L)
print(m,b)

plt.scatter(data.sqft_living,data.price, color="black")
plt.plot(list(range(0,500000)), [ m * x + b for x in range(0, 500000)], color="red")