from subprocess import check_output
import numpy as np 
import pandas as pd 
import matplotlib.pyplot as plt 

#importando base de dados
base_de_dados = pd.read_csv("kc_house_data.csv")
tamanho_da_casa = base_de_dados['sqft_living']
valor = base_de_dados['price']

x = np.array(tamanho_da_casa).reshape(-1, 1)
y = np.array(valor)


#Dividindo a base dados em treino e teste
from sklearn.model_selection import train_test_split
treinoX, testeX, treinoY, testeY = train_test_split(x,y,test_size=1/3, random_state=0)

#Criando a regressão linear
from sklearn.linear_model import LinearRegression 
regressor = LinearRegression()
regressor.fit(treinoX, treinoY)

r_squared = regressor.score(x, y)
print(f'Coeficiente de Determinação: {r_squared}')


#Visualizando o restultado dos treinos
plt.scatter(treinoX, treinoY, color= 'red')
plt.plot(treinoX, regressor.predict(treinoX), color = 'blue')
plt.title ("Visualizador do treino")
plt.xlabel("Tamanho da casa")
plt.ylabel("Valor")
plt.show()

#Visualizando o restultado dos testes
plt.scatter(testeX, testeY, color= 'red')
plt.plot(treinoX, regressor.predict(treinoX), color = 'blue')
plt.title("Visualizador do teste")
plt.xlabel("Tamanho da casa")
plt.ylabel("Valor")
plt.show()