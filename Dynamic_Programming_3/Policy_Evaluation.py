import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import argparse
from tqdm import tqdm
import warnings
warnings.filterwarnings('ignore')


#Gamma'yı belirleme
parser=argparse.ArgumentParser()
parser.add_argument("--gamma",type=float,default=0.9)
args=parser.parse_args()



#Adım,sonraki durum ve ödülü döndürme

def env_step(state:tuple , act: tuple )->tuple:

    """
    state: o anki durum (x(satır),y(sütun))
    act : hareket (d_row(delta-row),d_col(delta-col)) aralarındaki farklar
    """

    assert state not in [(0,0),(3,3)] , "Terminal durumda hareket edemezsiniz."

    #Hareketi güncelleme
    next_row = state[0] + act[0]
    next_col = state[1] + act[1]


    
    if (0<= next_row < 4) and (0<= next_col < 4):
        return -1, (next_row, next_col) 
    
    #Eğer durum dışına çıkarsa
    else:
        return -1, state #yeni durum eski durumda kalsın ve -1 ödül alsın.
    



#Iterative Policy Evaluation (Yinemeli Politika Değerlendirme)
def policy_evaluation(grid_world:np.ndarray,
                        actions:list,
                        probs:list,
                        gamma:float,
                        delta:float,
                        terminal_states:list) -> tuple:
    
    """
    grid: durumlar için ödül matrisi
    probs: eylemler için olasılık dağılımı
    terminal_states: bitiş durumları
    gamma: indirim faktörü
    theta: durdurma kriteri
    """

    #İki Dizi (Two-list) Yöntemi
    Q= np.zeros(shape=(grid_world.shape[0],grid_world.shape[1],len(actions)))

    #Tüm durumlar için döngü

    for row in range(4):
        for col in range(4):

            #Eğer bitme durumda ise devam et.
            if (row,col) in terminal_states:
                
                continue
            
            #Tüm eylemler için döngü
            for index,act in enumerate(actions):

                #Ödülü - sonra durumu al
                reward , next_state = env_step((row,col),act)

                #Sonraki durumun değerini al
                next_value= grid_world[next_state]

                #Q değerini güncelleme
                Q[row, col, index] = probs[index] * (reward + gamma * next_value)

    #İki dizi yöntemi ile Q değerlerini hesapladıktan sonra grid dünyasını güncelleme
    new_grid_world= Q.sum(axis=-1) #Son eksen boyunca toplama
    max_diff = np.abs(new_grid_world - grid_world).max() #Maksimum farkı bulma

    return new_grid_world , max(max_diff,theta) #Yeni grid dünyasını ve maksimum farkı döndürme

#Greedy Policy Improvement 
def greedy_policy(grid_world:np.ndarray,
                    actions:list,
                    probs:list,
                    gamma:float,
                    terminal_states:list) -> list:
    
    Q= np.zeros(shape=(grid_world.shape[0],grid_world.shape[1],len(actions)))

    #Eylemler için karakter listesi
    act_list = np.array([0x2190, 0x2193, 0x2192, 0x2191])


   
    greedy_acts= []

    #Tüm durumlar için döngü
    for row in range(grid_world.shape[0]):
        acts_row = [] #Her satır için eylemleri saklamak için liste oluştur

        for col in range(grid_world.shape[1]):

            #Eğer bitiş durumda ise devam et
            if (row,col) in terminal_states:
                acts_row.append('')
                continue

            #Tüm eylemler için döngü
            for i, act in enumerate(actions):

                #Ödülü - sonra durumu al
                reward , next_state = env_step((row,col),act)

                #Sonraki durumun değerini al
                next_value= grid_world[next_state]

                #Q değerini güncelleme
                Q[row, col, i] = probs[i] * (reward + gamma * next_value)

            #greedy_sels: Q değerlerinden en büyüğünün indeksini al
            #np.abs= np.absolute: mutlak değer 
            greedy_sels = np.where(np.abs(Q[row, col] - Q[row, col].max()) < 0.0001)[0]
            #Eylemleri karakterlere dönüştürün ve bunları bir dizede birleştirin
            acts = "".join([chr(act_list[i]) for i in greedy_sels])
            

            #Eylemleri saklamak için listeye ekle
            acts_row.append(acts)
        
        #Eylemleri saklamak için listeyi büyük listeye ekle
        greedy_acts.append(acts_row)
        

    return greedy_acts



#Görselleştirme
def plot_heatmap(data: np.ndarray, 
                 annotations: list,
                 axes: np.ndarray,
                 curr_row: int,
                 cbar: bool = False,
                 fmt: str = '.1f',
                 cmap: str = 'Blues',
                 linewidths: float = 0.5) -> None:
    """
    Plots a heatmap for a given 2-D array of data and its annotations on specified axes.

    Parameters:
    - data (np.ndarray): A 2-D array of numerical data.
    - annotations (list): A list of annotations to display on the action heatmap.
    - axes (np.ndarray): An array of matplotlib axes on which to plot the heatmaps.
    - curr_row (int): The current row index in the axes array to plot the heatmaps.
    - cbar (bool): Whether to display a color bar. Defaults to False.
    - fmt (str): The string format for numeric annotations. Defaults to '.1f'.
    - cmap (str): The colormap to use. Defaults to 'Blues'.
    - linewidths (float): The width of the lines that will divide each cell. Defaults to 0.5.
    """
    assert len(data.shape) == 2, 'Input must be a 2-D array'
    ax_value = axes[curr_row, 0]  # Axis for plotting values
    ax_action = axes[curr_row, 1]  # Axis for plotting actions

    title = f'iteration={curr_row}' if curr_row != -1 else 'iteration=$\infty$'

    # Plot values heatmap
    sns.heatmap(data, ax=ax_value, cbar=cbar, fmt=fmt, annot=True, cmap=cmap, linewidths=linewidths)
    ax_value.set_yticks([])
    ax_value.set_xticks([])
    ax_value.set_title(title + ' (Values)', fontweight='bold')
    

    # Plot actions heatmap
    sns.heatmap(data, ax=ax_action, cbar=cbar, annot=annotations, fmt='', cmap=cmap, linewidths=linewidths)
    ax_action.set_yticks([])
    ax_action.set_xticks([])
    ax_action.set_title(title + ' (Actions)', fontweight='bold')
    





if __name__ == '__main__':
     
    #GridWorld'ü oluştur
    grid_world = np.zeros(shape=(4,4))
    actions = [(0, -1), (1, 0), (0, 1), (-1, 0)]

    gamma = args.gamma #İndirim faktörü
    theta= 0.0001 #Durdurma kriteri


    #Rastgele politika için eşit olasılıklı aksiyonlar
    probs = [1. / len(actions)] * len(actions)

    #Bitiş durumları
    terminal_states = [(0,0),(3,3)]


    #sonuçları saklamak için figür ve eksenler oluştur
    fig,axes = plt.subplots(6,2,figsize=(10,20))
    
    #Satır sayacı
    curr_row = 0

    
    #Döngüye gir ve theta isteniken değerden düşük olana kadar devam et.
    for i in tqdm(range(100)):


        #Belirli iterasyonlarda ilgili heatmap'ı görselleştirme
        if i in  [0,1,2,3,10]:

            #Greedy politikayı hesapla
            greedy_acts = greedy_policy(grid_world, 
                                        actions, 
                                        probs, 
                                        gamma, 
                                        terminal_states)
            
            #Heatmap'ları görselleştir
            plot_heatmap(grid_world,greedy_acts,axes,curr_row)
    
            curr_row += 1

        
        delta=0 #reset delta

        #Politika değerlendirmeyi başlat
        grid_world,delta = policy_evaluation(
                            grid_world, 
                            actions, 
                            probs,
                            gamma,
                            delta, 
                            terminal_states)

        if delta < theta:
            break

        greedy_acts = greedy_policy(grid_world, 
                                actions, 
                                probs, 
                                gamma, 
                                terminal_states)
        

        # #Ajanın satırlarda hangi aksiyonları aldığını göster
        # for row, row_acts in enumerate(greedy_acts):
        #     print(f'{row} : {row_acts}')

        plot_heatmap(grid_world, greedy_acts, axes, -1)
        plt.tight_layout()
        plt.subplots_adjust(left=0.1, right=0.9, top=0.9, bottom=0.1, hspace=0.4, wspace=0.2)
        plt.savefig('example_4_1.png')