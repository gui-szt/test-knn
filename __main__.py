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

def Treino(a: int, b: int, dl: pd.DataFrame):
    global Pares
    Pares = []

    k = a
    r = b

    # Adiciona nomes às colunas
    dl.columns = ["sepal_length", "sepal_width", "petal_length", "petal_width", "class"]

    # Remove linhas vazias
    dl = dl.dropna()

    #Alaeatoriza os pares
    al.aleatorio(len(dl) - 1, r)

    X = dl[["sepal_length","sepal_width","petal_length","petal_width"]].to_numpy()

    for pair_idx in range(int(r)):
        Acu = []
        comp_list = al.C[pair_idx]
        base_list = al.B[pair_idx]

        for w in range(int(k)):  # K vizinhos

            erro = 0

            for idx_c in comp_list:
                print(f"Processando par {pair_idx + 1}, K={w + 1}, elemento de teste {idx_c}...")
                distances = []
                real = dl.loc[int(idx_c), "class"]

                # calcular distâncias
                for idx_b in range(len(base_list)):
                    d = np.linalg.norm(X[idx_c] - X[idx_b])
                    distances.append(float(d))

                # extrair w+1 vizinhos mais próximos
                supos = []
                dist_tmp = distances.copy()

                for n in range(int(w) + 1):
                    pos = dist_tmp.index(min(dist_tmp))#index da menor distância
                    supos.append(dl.loc[base_list[pos], "class"])#guardar a suposição
                    dist_tmp[pos] = float('inf')# marcar como já usado

                # moda seguro
                try:
                    mode = statistics.mode(supos)
                except:
                    mode = statistics.multimode(supos)[0]

                if mode == real:
                    erro += 1

            # Acurácia do par para este K
            Acu.append(round(float(erro / len(comp_list)), 4))

        Pares.append(par(comp_list, base_list, Acu))

    # Gerar tabela final
    list_acu = [p.acu for p in Pares]
    df_acc = pd.DataFrame(list_acu)
    df_acc.columns = [f"k={i+1}" for i in range(df_acc.shape[1])]
    df_acc.index = range(1, len(df_acc) + 1)

    df_acc.to_csv("precision_pairs.csv", index_label="Par")

    with open("precisoes.txt", "w", encoding="utf-8") as f:
        f.write("PARES:\n\n")
        for i in range(len(Pares)):
            f.write(f"PAR {i+1}:\n")
            f.write(f"Vetor comparação: {Pares[i].comp}\n")
            f.write(f"Vetor base: {Pares[i].teste}\n")
            f.write(f"Acurácias: {Pares[i].acu}\n\n")


def main() -> None:

    df = pd.read_csv("bezdekIris.data", header=None)
    df.columns = ["sepal_length", "sepal_width", "petal_length", "petal_width", "class"]
    X = df[["sepal_length","sepal_width","petal_length","petal_width"]].to_numpy()

    try:
        pres = pd.read_csv("precision_pairs.csv", index_col="Par")
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
                numeric_cols = df.select_dtypes(include=np.number).columns
                Val = pd.DataFrame(index=linhas, columns=numeric_cols)

                for col in numeric_cols:
                    col_values = df[col]
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
                Val=Val.astype(float)

                for i in numeric_cols:
                    fig, axs = plt.subplots(8, 8, figsize=(10, 7))

                    axs[0, 0].scatter(Val.iloc[i], y_main)
                    axs[0, 0].set_title("main")
                    axs[0, 0].grid(True)

                    # Ajuste de layout
                    plt.tight_layout()
                    plt.title(i)
                    plt.show()

            case "t":
                k = int(input("Digite o número de vizinhos: "))
                r = int(input("Digite o número de pares: "))
                Treino(k, r, df)

            case "c":
                if acur == 0:
                    print("\nTreine primeiro.\n")
                    continue

                Novo = []
                dist_suposi = []

                for i in df.columns:
                    if i != "class":
                        n = float(input(f"\n{i}: "))
                        Novo.append(n)

                Novo = np.array(Novo)

                key = int(input("\nk vizinhos a considerar: "))
                if key > acur:
                    Treino(key, 15, df)

                #selecionar o par com maior acurácia para o k escolhido
                coluna = pres.iloc[:, key].tolist()
                coluna = [float(c) for c in coluna]
                print("\n {coluna}\n")

                f = max(coluna)

                idx = pr.loc[coluna.index(f) + 1, "Base_index"]

                idx = idx.strip("[]").split(",")
                idx = [i.strip() for i in idx]

                for i in range(len(idx)):
                    d = np.linalg.norm(Novo - X[int(idx[i])])
                    dist_suposi.append(float(d))

                #lista das distancias dos k vizinhos mais próximos do novo objeto
                sup=[]
                for i in range(int(key)):
                    pos=dist_suposi.index(min(dist_suposi))#index da menor distância
                    sup.append(df.loc[int(idx[pos]), "class"])#guardar a suposição
                    dist_suposi[pos]=float('inf')# marcar como já usado

                print("\nClasse estimada:", statistics.mode(sup))

            case "r":
                medias=[]
                variancias=[]
                desviom=[]
                p=0
                while p < acur:
                    col=pres.iloc[:,p+1].tolist()
                    col=[float(c) for c in col]
                    col=np.array(col)
                    medias.append(round(statistics.mean(col),4))
                    variancias.append(round(statistics.variance(col),4))
                    desviom.append((abs(col - col.mean())).mean())
                    p+=1
                x = [i+1 for i in range(acur)]

                fig, axs = plt.subplots(1, 3, figsize=(12, 4))

                axs[0].scatter(x, medias)
                axs[0].set_title("Médias")
                axs[0].set_xticks(x)

                for xi, yi in zip(x, medias):
                    axs[0].text(xi, yi, f"{yi:.4f}", ha='center', va='bottom', fontsize=7)

                axs[1].scatter(x, variancias)
                axs[1].set_title("Variâncias")
                axs[1].set_xticks(x)

                for xi, yi in zip(x, variancias):
                    axs[1].text(xi, yi, f"{yi:.4f}", ha='center', va='bottom', fontsize=7)

                axs[2].scatter(x, desviom)
                axs[2].set_title("Desvio Médio Absoluto")
                axs[2].set_xticks(x)

                for xi, yi in zip(x, desviom):
                    axs[2].text(xi, yi, f"{yi:.4f}", ha='center', va='bottom', fontsize=7)

                plt.tight_layout()
                plt.show()


            case "s":
                menu = "s"
                break

            case _:
                print("Opcao invalida\n")

if __name__ == "__main__":
    main()
