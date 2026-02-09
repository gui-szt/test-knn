import pandas as pd
import aleat as al 
import matplotlib.pyplot as plt
import numpy as np
import statistics
from scipy.stats import skew, kurtosis


class par:
    def __init__(self, comp, teste, acu):
        self.comp = comp
        self.teste = teste
        self.acu = acu

Pares = []  # Todos os pares de vetores

def MinMax(a: pd.DataFrame):
    df_norm = a.copy()

    for col in df_norm.columns:
        if col != "class" and pd.api.types.is_numeric_dtype(df_norm[col]):
            min_x = df_norm[col].min()
            max_x = df_norm[col].max()
            if min_x != max_x:
                df_norm[col] = (df_norm[col] - min_x) / (max_x - min_x)

    return df_norm

def Treino(a: int, b: int, dl: pd.DataFrame, X: np.ndarray ) :
    global Pares
    Pares = []

    k = a
    r = b

    #Alaeatoriza os pares
    B,C=al.aleatorio(len(dl) - 1, r)
    classes = dl["class"].unique()

    for pair_idx in range(int(r)):
        Acu = []
        comp_list = B[pair_idx]
        base_list = C[pair_idx]
        print(f"\nTreinando par {pair_idx + 1}...")
        
        for w in range(int(k)):  # K vizinhos

            metrics = {
                c: {
                    "VP": 0,
                    "FP": 0,
                    "FN": 0,
                    "VN": 0,
                    "accuracy": 0.0,
                    "precision": 0.0,
                    "recall": 0.0,
                    "f1": 0.0
                }
                for c in classes
            } #VAriaveis da matrix de confusão para calcular precisão, revocação e f1-score


            for idx_c in base_list:
               
                distances = []
                real = dl.loc[int(idx_c), "class"]

                # calcular distâncias
                for idx_b in comp_list:
                    d = np.linalg.norm(X[idx_c] - X[idx_b])
                    distances.append(float(d))
                    

                # extrair w+1 vizinhos mais próximos
                supos = []
                dist_tmp = distances.copy()

                for n in range(int(w) + 1):
                    pos = dist_tmp.index(min(dist_tmp))#index da menor distância
                    supos.append(dl.loc[comp_list[pos], "class"])#guardar a suposição
                    dist_tmp[pos] = float('inf')# marcar como já usado

                # moda seguro
                try:
                    mode = statistics.mode(supos)
                except:
                    mode = statistics.multimode(supos)[0]

                for c in metrics:
                    if real == c and mode == c:
                        metrics[c]["VP"] += 1
                    elif real != c and mode == c:
                        metrics[c]["FP"] += 1
                    elif real == c and mode != c:
                        metrics[c]["FN"] += 1
                    else:
                        metrics[c]["VN"] += 1
            

            # Calcular métricas do par para este cada classe e para esse K
            for c in classes:
                VP = metrics[c]["VP"]
                FP = metrics[c]["FP"]   
                FN = metrics[c]["FN"]
                VN = metrics[c]["VN"]
                total=VP + FN + FP + VN
                metrics[c]["accuracy"] = (VP + VN) / total if total > 0 else 0
                metrics[c]["precision"] = VP / (VP + FP) if (VP + FP) > 0 else 0
                metrics[c]["recall"] = VP / (VP + FN) if (VP + FN) > 0 else 0

                p = metrics[c]["precision"]
                r = metrics[c]["recall"]
                metrics[c]["f1"] = 2 * p * r / (p + r) if (p + r) > 0 else 0

            # Média macro das métricas
            macro = {
                "precision": sum(m["precision"] for m in metrics.values()) / len(metrics),
                "recall": sum(m["recall"] for m in metrics.values()) / len(metrics),
                "f1": sum(m["f1"] for m in metrics.values()) / len(metrics)
            }
            acertos = 0
            total = 0

            for c in classes:
                acertos += metrics[c]["VP"]
                total += metrics[c]["VP"] + metrics[c]["FN"]

            acc = acertos / total if total > 0 else 0
            Acu.append(round(acc, 4))

        
        Pares.append(par(comp_list, base_list, Acu))

    # Gerar tabela final
    list_acu = [p.acu for p in Pares]
    df_acc = pd.DataFrame(list_acu)
    df_acc.columns = [f"k={i+1}" for i in range(df_acc.shape[1])]
    df_acc.index = range(1, len(df_acc) + 1)

    df_acc.to_csv("acuracy_pairs.csv", index_label="Par")

    with open("precisoes.txt", "w", encoding="utf-8") as f:
        f.write("PARES:\n\n")
        for i in range(len(Pares)):
            f.write(f"PAR {i+1}:\n")
            f.write(f"Vetor comparação: {Pares[i].comp}\n")
            f.write(f"Vetor base: {Pares[i].teste}\n")
            f.write(f"Acurácias: {Pares[i].acu}\n\n")


def main() -> None:

    df,dl,X,opc = None,None,None,None
    
    match input("Escolha o conjunto de dados:\n\n1 - Iris\n2 - Breast Cancer Wisconsin\n\nOpção: "):
        case "1":
            df = pd.read_csv("bezdekIris.data", header=None)
            df.columns = ["sepal_length", "sepal_width", "petal_length", "petal_width", "class"]
            dl=df.copy()#Copia do original para analise estatistica 
            df=MinMax(df)#Normaliza os dados
            X = df[["sepal_length","sepal_width","petal_length","petal_width"]].to_numpy()
        case "2":
            df = pd.read_csv("data.csv", header=None)
            df = df.iloc[:, :-1]#elimina a coluna vazia no final do arquivo original

            df.columns = [
                "id", "class",
                "radius_mean", "texture_mean", "perimeter_mean", "area_mean",
                "smoothness_mean", "compactness_mean", "concavity_mean",
                "concave_points_mean", "symmetry_mean", "fractal_dimension_mean",
                "radius_se", "texture_se", "perimeter_se", "area_se",
                "smoothness_se", "compactness_se", "concavity_se",
                "concave_points_se", "symmetry_se", "fractal_dimension_se",
                "radius_worst", "texture_worst", "perimeter_worst", "area_worst",
                "smoothness_worst", "compactness_worst", "concavity_worst",
                "concave_points_worst", "symmetry_worst", "fractal_dimension_worst"
            ]
            df["class"] = df["class"].astype("category")
            for col in df.columns:
                if col != "class":
                    df[col] = pd.to_numeric(df[col], errors="coerce")
            
            df = df.drop(columns=['id'])
            dl=df.copy()#Copia do original para analise estatistica 
            df=MinMax(df)#Normaliza os dados
            df.to_csv("df.csv", index_label="Dados")
            X = df[["radius_mean","texture_mean","perimeter_mean","area_mean","smoothness_mean","compactness_mean","concavity_mean","concave_points_mean","symmetry_mean","fractal_dimension_mean","radius_se","texture_se","perimeter_se","area_se","smoothness_se","compactness_se","concavity_se","concave_points_se","symmetry_se","fractal_dimension_se","radius_worst","texture_worst","perimeter_worst","area_worst","smoothness_worst","compactness_worst","concavity_worst","concave_points_worst","symmetry_worst","fractal_dimension_worst"]].to_numpy()
        case _:
            print("Opção inválida.")        

    try:
        pres = pd.read_csv("acuracy_pairs.csv", index_col="Par")
        acur = len(pres.columns) - 1
        print(acur)
    except:
        print("\nAinda não existe arquivo de treino. Execute o treino primeiro.\n")
        acur = 0

    try:
        pr = pd.read_csv("vetores_pairs.csv", header=None)
        pr.columns = ["Par","Base_index","Teste_index"]
    except:
        print("\nAviso: arquivo vetores_pairs.csv não encontrado.\n")
        pr = None


    menu = "x"

    while menu != "s":
        menu = input("\nDigite\n\nT - Treino\nC - Classificacao\nD - Analise dos dados\nR - Analise dos resultados\nS - Sair\nOpcao: ").lower()
        match menu:
            case "d":
                # Criar tabela com métricas
                linhas = [
                    'Media',
                    'Mediana',
                    'Variancia',
                    'Desvio padrao',
                    'Desvio medio absoluto',
                    'Desvio mediano absoluto',
                    'Obliquidade',
                    'Curtose'
                ]

                # Selecionar apenas colunas numéricas
                numeric_cols = [c for c in df.columns if c != "class"]
                Val = pd.DataFrame(index=linhas, columns=numeric_cols)

                for col in numeric_cols:
                    col_values = dl[col]
                    Val.loc['Media', col] = col_values.mean()
                    Val.loc['Mediana', col] = col_values.median()
                    Val.loc['Variancia', col] = col_values.var()
                    Val.loc['Desvio padrao', col] = col_values.std()
                    Val.loc['Desvio medio absoluto', col] = (abs(col_values - col_values.mean())).mean()
                    Val.loc['Desvio mediano absoluto', col] = (abs(col_values - col_values.median())).median()
                    Val.loc['Obliquidade', col] = skew(col_values)
                    Val.loc['Curtose', col] = kurtosis(col_values)

                print("\nTabela completa das estatísticas:")
                print(Val)

                #COVARIANCIA
                print("\nMatriz de Covariancia:")
                cov_matrix = dl[numeric_cols].cov()     
                print(cov_matrix)

                #CCORRELACAO
                print("\nMatriz de Correlaçao:")
                corr_matrix = dl[numeric_cols].corr()     
                print(corr_matrix)

                #GRAFICOS DE DISTRIBUIÇÃO por classe

                classes = dl["class"].unique()
                cols = [c for c in dl.columns if c != "class"]

                rows = int(np.ceil(len(cols) / 4))
                fig, axs = plt.subplots(rows, 4, figsize=(20, 5 * rows))
                axs = axs.flatten()

                for j, col in enumerate(cols):
                    dados = [dl[dl["class"] == c][col] for c in classes]
                    axs[j].boxplot(dados, labels=classes)
                    axs[j].set_title(col)
                    axs[j].tick_params(axis='x', rotation=45)

                plt.tight_layout()
                plt.show()


                #GRAFICOS DE RELAÇAO
                # BOXPLOT GERAL DAS VARIÁVEIS
                plt.figure(figsize=(18, 8))
                dl[cols].boxplot(rot=90)
                plt.title("Distribuição das variáveis (Boxplot)")
                plt.tight_layout()
                plt.show()


            case "t":
                k = int(input("Digite o número de vizinhos: "))
                r = int(input("Digite o número de pares: "))
                Treino(k, r, df, X)

            case "c":

                Novo = []
                dist_suposi = []

                for i in df.columns:
                    if i != "class":
                        n = float(input(f"\n{i}: "))
                        Novo.append(n)
                
                #NORMALIZAR O NOVO VETOR
                copy_dl=dl.copy()
                copy_dl.loc[len(copy_dl)] = Novo + [""] #adicionar o novo vetor ao dataframe original   
                copy_dl=MinMax(copy_dl)
                Novo=copy_dl.iloc[-1, :-1].tolist() #obter o vetor normalizado
                Novo = np.array(Novo)
                print("\nVetor normalizado:", Novo)
                medi=[]
                for k in range(acur+1):
                    colu = pres.iloc[:, k ].tolist()
                    media=0
                    for i in colu:
                        media+=float(i)
                    medi.append(round(float(media/len(colu)),4))

                key = medi.index(max(medi)) + 1  #k escolhido baseado na posiçao da maior acurácia média
                print(key)
                #selecionar o par com maior acurácia para o k escolhido

                for i in df.index:
                    d = np.linalg.norm(Novo - X[i])
                    print(X[i])
                    print(f"Distancia do novo objeto ao elemento {i}: {d}")

                    dist_suposi.append(float(d))

                #lista das distancias dos k vizinhos mais próximos do novo objeto
                sup=[]
                for i in range(int(key)):
                    pos=dist_suposi.index(min(dist_suposi))#index da menor distância
                    sup.append(df.loc[int(pos), "class"])#guardar a suposição
                    dist_suposi[pos]=float('inf')# marcar como já usado

                print("\nClasse estimada:", statistics.mode(sup))

            case "r":
                #Analisar as precisoes para k vizinhos proximos
                medias = []
                for key in range(acur+1):
                    coluna = pres.iloc[:, key ].tolist()
                    media=0
                    for i in coluna:
                        media+=float(i)
                    medias.append(round(float(media/len(coluna)),4))
                    print(f"\nPrecisão média para k={key + 1}: {medias[key]}")
                
                plt.plot(range(1, acur + 2), medias,'o')
                for i_x, i_y in zip(range(1, acur + 2), medias):
                    plt.text(i_x, i_y, '({}, {})'.format(i_x, i_y), ha='center', va='top')

                plt.show()

                # BOXPLOT DAS PRECISÕES POR k
                dados_box = [pres.iloc[:, k].astype(float) for k in range(acur + 1)]

                plt.figure(figsize=(10, 6))
                plt.boxplot(dados_box, labels=[f"k={i+1}" for i in range(acur + 1)])
                plt.ylabel("Precisão")
                plt.title("Distribuição das precisões por número de vizinhos (k)")
                plt.grid(True, axis='y')
                plt.show()

                

            case "s":
                menu = "s"
                break

            case _:
                print("Opcao invalida\n")

if __name__ == "__main__":
    main()
