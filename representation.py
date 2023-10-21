import numpy as np
import matplotlib.pyplot as plt




def graph(list_of_stock,list_of_names):
    all_mean_arr=[]
    all_var_arr=[]
    keys=list_of_stock[0][0].keys()
    labels=[]

    nb_labels = len(labels)
    for key in keys:
        labels.append(key)
    for stock in list_of_stock:
        L=[]
        for key in labels:
            L.append([stock[i][key] for i in range(len(stock))])
        arr=np.array(L)
        # before ax 0 : (nb of key) ax 1 : (nb of simulation) ax 2 : (nb of epoch)
        mean_arr = np.average(arr,axis=1, keepdims=False)
        var_arr = np.var(arr, axis=1, keepdims=False)
        all_mean_arr.append(mean_arr)
        all_var_arr.append(var_arr)
    
    nb_epoch = all_mean_arr[0].shape[1]
    # The new axis are, ax 0 : (nb of key) ax 1 : (nb of epoch) ax 2 : (nb different tests to compare)
    final_mean_arr = np.stack(all_mean_arr, axis=2)
    final_var_arr = np.stack(all_var_arr, axis=2)
    plt.figure()
    for k,key in enumerate(labels):
        print(key)
        plt.subplot( 2 , 2, k+1)
        for i,name in enumerate(list_of_names):
            plt.plot(range(1, nb_epoch+1), final_mean_arr[k,:,i], label=name)
            plt.fill_between(range(1, nb_epoch+1), final_mean_arr[k,:,i], final_var_arr[k,:,i])
        plt.xlabel('Epoch')
        plt.ylabel(key)
        plt.legend(loc='lower right')
    plt.tight_layout()
    plt.show()

def test_graph():
    """This function is used to test the graph function.
    """
    test1= [ {'test': [1,10],'loss': [50,10000]},{'test': [2,5],'loss': [50,-10000]},
           {'test': [3,5],'loss': [1,80]},{'test': [4,5],'loss': [1,8000]} ]
    test2 = [ {'test': [1,10000],'loss': [1,10000]},{'test': [2,5],'loss': [1,10000]},
           {'test': [3,5],'loss': [1,10000]},{'test': [4,5],'loss': [1,10000]} ]
    test3 = [ {'test': [1,10000],'loss': [1,10000]},{'test': [2,5],'loss': [1,10000]},
           {'test': [3,5],'loss': [1,10000]},{'test': [4,5],'loss': [1,10000]} ]
    graph([test1,test2,test3], ['test1','test2','test3'])
if __name__ == '__main__':
    test_graph()