import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import f1_score,recall_score,precision_score, confusion_matrix

def curve(model,x_train,y_train,qte=100):

    limiares = np.linspace(0,1,qte)

    recalls = []
    precisions =[]
    f1s = []
    custo_mitigar = []
    rendimentos =[]
    rendimentos_n_contabilizados = []
    gastos_list = []

    for limiar in limiares:

        
        pred = model.predict_proba(x_train)[:,1]

        y_pred = np.where(pred >= limiar,1,0)

        matrix = confusion_matrix(y_train, y_pred)
        tn, fp, fn, tp = matrix.ravel()
        custo_mitigar.append(tp*55)
        rendimentos.append(tn*55)
        rendimentos_n_contabilizados.append(fp*55)
        gastos_list.append(fn*55)

        recalls.append(recall_score(y_train,y_pred))
        precisions.append(precision_score(y_train,y_pred))
        f1s.append(f1_score(y_train,y_pred))


    fig, ax = plt.subplots(1,2,figsize=(15,7))

    ax[0].plot(limiares,precisions,label='Precision')
    ax[0].plot(limiares,recalls,label='Recall')
    ax[0].plot(limiares,f1s,label='F1')
    ax[0].set_xlabel('Limiar')
    ax[0].set_ylabel('(%)')
    ax[0].legend()

    ax[1].plot(rendimentos_n_contabilizados,custo_mitigar)
    ax[1].set_xlabel('rendimentos não contabilizados')
    ax[1].set_ylabel('Custos a mitigar')

    plt.tight_layout()
    return limiares, recalls, precisions, f1s
