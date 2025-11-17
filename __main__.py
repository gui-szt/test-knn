import pandas as pd
import aleat as al
#import matplotlib.pyplot as plt
import numpy as np
import statistics

class par:
    def __init__(self, comp, teste,acu):
        self.comp = comp
        self.teste = teste
        self.acu=acu
#PRECISO FAZER AS MEDIDAS SEPARADAS PARA CADA K
Pares=[] #Todos os pares de vetores

def Treino(a : int ,b : int):
	k=a
	r=b
	# Lê o arquivo (ajuste o nome conforme o seu)
	df = pd.read_csv("bezdekIris.data", header=None)

	# Adiciona nomes às colunas
	df.columns = ["sepal_length", "sepal_width", "petal_length", "petal_width", "class"]

	# Remove linhas vazias (às vezes há uma no final)
	df = df.dropna()

	al.aleatorio(len(df)-1, r)

	X = df[["sepal_length","sepal_width","petal_length","petal_width"]].to_numpy()
	for pair_idx in range(int(r)):
		# C e B sao matrizes "Pares",cada C[i] teste existe um B[i] comparação
		Acu=[] #Todas as porcentagens de acertos por pares de vetores
					
		for w in range(int(k)): #Fazer varios teste para w ate k

			comp_list = al.C[pair_idx]   
			base_list = al.B[pair_idx]   
			erro=0 #Taxa deee erro do par

			for idx_c in range(len(comp_list)):
				distances = []
				supos=[] #SUPOSIÇÃO BASEADA NA MAIS PROXIMA
				real=df.loc[int(comp_list[idx_c]), "class"] #CLASSE REAL DO ITEM C[IDX_C]
							
				for idx_b in range(len(base_list)):
					# distância euclidiana entre as features
					d = np.linalg.norm(X[comp_list[idx_c]] - X[base_list[idx_b]])
					distances.append(float(d))
					print(f"Distância entre C[{comp_list[idx_c]}] e B[{base_list[idx_b]}]: {d}")

					#SUUPOSIÇAO K VEZES
					for n in range(int(w)+1):
						for i in range(int(len(distances))):
							if float(min(distances))==float(distances[i]):
								supos.append(df.loc[int(i), "class"])
								distances[i]=10
					#try:
					mode= statistics.mode(supos)
					if real != mode:
						erro +=1
					# except:
						#    multi=statistics.multimode(supos)			
		#MEDIDA DE ACURACIA DO PAR
			Acu.append(round(float((len(comp_list) - erro)/len(comp_list)), 4))
		Pares.append(par(comp_list, base_list,Acu))
	list_acu=[]
	for p in Pares:
		list_acu.append(p.acu)                    # lista de listas 
	df_acc = pd.DataFrame(list_acu)                      # shape (n_pares, k)
	df_acc.columns = [f"k={i+1}" for i in range(df_acc.shape[1])]  # nomeia colunas
	df_acc.index = range(1, len(df_acc) + 1)             # index 1-based para os pares

	# salvar em CSV
	df_acc.to_csv("precision_pairs.csv", index_label="Par")
	with open("precisoes.txt", "w", encoding="utf-8") as f:
		f.write("PARES:" + "\n\n")
		for i in range(len(Pares)):
			f.write("PAR "+ str(i+1) + ":"  + "\n")
			f.write("Vetor comparação: " + str(Pares[i].comp)  + "\n")
			f.write("Vetor base: " + str(Pares[i].teste)  + "\n")
			f.write("Acurácias: " + str(Pares[i].acu)  + "\n\n")


def main() -> None:
	pres = pd.read_csv("precision_pairs.csv", index_col="Par")
	acur=len(pres.columns)
    

	menu="x"
        
	while menu !="s":
		k=acur
		menu=input("\ndigite\n\nT-Treino\nC-Classificacao\nS-Sair\nOpcao: ")
		match menu:
			case "t":
				k=input("Digite o número de vizinhos a serem considerados: ")
				r=input("Digite o número de vetores aleatórios a serem gerados: ")
				Treino(k,r)
				
			case "c":
				df = pd.read_csv("bezdekIris.data", header=None)
				df.columns = ["sepal_length", "sepal_width", "petal_length", "petal_width", "class"]
				X = df[["sepal_length","sepal_width","petal_length","petal_width"]].to_numpy()
				
				pr = pd.read_csv("vetores_pairs.csv", header=None)
				pr.columns = ["Par","Base_index","Teste_index"]
				#P =pr[ ["Par","Base_index","Teste_index"]].to_numpy()
				
				
				Novo=[0,0,0,0]#Objeto a ser classificado
				suposi=[]#Todas as classes de k
				dist_suposi=[]


				Novo[0]=float(input("\nsepal_length:"))
				Novo[1]=float(input("\nsepal_width:"))
				Novo[2]=float(input("\npetal_length:"))
				Novo[3]=float(input("\npetal_width:"))
				
				key=input("\nVizinhos proximos a se considerar:")
				if (int(key)>acur):
					Treino(int(key), 100)
				idx=[]
				coluna = pres.iloc[: ,(int(key))].tolist()
				print(coluna)
				for i in range(int(len(coluna))):
					coluna[i]=float(coluna[i])
				
				f=float(max(coluna))
				for i in range(int(len(coluna))):
					if f==float(coluna[i]):
						idx=pr.loc[i+1, "Base_index"] #index do modelo
						
				# Remove colchetes
				idx =idx.strip("[]")

				# Divide pelos números separados por vírgula
				idx = idx.split(",")

				# Remove espaços extras
				idx = [i.strip() for i in idx]

				for i in range(len(idx)):#distancia de novo para todos os vetores do modelo
					d= np.linalg.norm(Novo - X[int(idx[i])])
					dist_suposi.append(float(d))
				
				for i in range(int(key)):
					dsup=dist_suposi.copy()
					for o in range(int(len(dist_suposi))):
						
						if float(min(dsup))==float(dsup[o]):
							print(df.loc[int(idx[i]), "class"])
							suposi.append(df.loc[int(o), "class"])
							dsup[int(idx[i])]=10
						
				print(statistics.mode(suposi))

				k = int(key)
				indices_k = np.argsort(dist_suposi)[:k]

				# Obtenha as classes dos vizinhos
				sup = [df.loc[int(idx[i]), "class"] for i in indices_k]

				# Imprimir a classe mais votada
				print(statistics.mode(sup))

				break
				
			case "s":
				break
			case _:
				print("Opcao invalida\n")
				break
            

                    

                
                
                

                


                    


   

                

    



if __name__ == "__main__":
    main()
# ...existing code...
