import random
import pandas as pd

B=[]  #Matrix d vetores de comparação
C=[]  #Matrix d vetores de teste
def aleatorio (p : int, r : int):
    a=int(p) + 1
    u=int(r)


    com=a*(7/10)
    tes=a - com
    A=[]
    for i in range(int (a)):
       A.append(i)
    
    A=sorted(A)
    
    b=0  #variavel de controle do laço de formaçaõa dos pares
    
    
    for b in range(u):
        j=0  #variavel de controle dos elementos de comparação
        c=0  #variavel de controle dos elementos de teste
        Comp=[]
        Teste=[]
        for i in range(com):
            Comp.append(int(-1))

        for i in range(tes):
            Teste.append(int(-1))

        #Fazer um vetor comp aleatorio
        while j<(com):
            erro=0  #teste pra saber se esse elemento aleatorio ja esta em B[b]
            num=random.randint(0,a-1)
            for o in range (com):
                if A[num]==Comp[o]:
                    erro=1
            if erro==0:
                Comp[j]=A[num]
                j+=1

        Cmp=sorted(Comp)
        if Cmp not in B:
            B.append(Cmp)
            # preencher Teste com elementos de A que não estão em Comp, até o tamanho desejado
            for y in range(a):
                if A[y] not in Cmp:
                    Teste[c]=(A[y])
                    c+=1
            C.append(Teste)
    
    # leitura/escrita de arquivo
    vetores=pd.DataFrame({"Base_index": [sorted(x) for x in B],
        "Teste_index": C
    })
    # index 1-based (opcional)
    vetores.index = range(1, len(vetores) + 1)

    # salvar em CSV
    vetores.to_csv("vetores_pairs.csv", index_label="Par")

    with open("example.txt", "w", encoding="utf-8") as f:
        f.write("Vetores de comparação:" + "\n")
        for i in range(u):
            f.write("Comparacao"+ str(i+1) + ":" + str(B[i]) + "\n")
        f.write("\nVetores de teste:\n")
        for i in range(u):
            f.write("Teste"+ str(i+1) + ":" + str(C[i]) + "\n")


