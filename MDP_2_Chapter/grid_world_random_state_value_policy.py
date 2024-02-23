import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from enum import Enum


#Hareketler için enum sınıfı
class ACTIONS(Enum):
    UP = (-1, 0)
    DOWN = (1, 0)
    LEFT = (0, -1)
    RIGHT = (0, 1)



def get_reward(state:tuple, act:tuple):
    """
    state: o anki durum (x(satır),y(sütun))

    act : hareket (d_row(delta-row),d_col(delta-col)) aralarındaki farklar


    output: reward and next state (ödülü ve bir sonraki durumu döndürür)
    """

    if state == (0,1):
        return 10, (4,1) # eğer durum (0,1) ise ödül 10 ve bir sonraki durum (4,1) olsun.
    
    if state == (0,3):
        return 5, (2,3) # eğer durum (0,3) ise ödül 5 ve bir sonraki durum (2,3) olsun. 
    
    #Hareketi güncelleme
    next_row = state[0] + act[0] 
    next_col = state[1] + act[1] 

   
    # Eğer bir sonraki durum sınırların dışına çıkarsa -1 ve aynı durumu döndür.

    if not(0 <= next_row < 5) or not(0 <= next_col < 5):
        return -1, state 
    
    # Eğer bir sonraki durum sınırların içindeyse 0 ve bir sonraki durumu döndür.
    return 0, (next_row, next_col) 



def value_function(grid_world:np.ndarray=np.zeros((5,5)),gamma:float=0.9):
    """
    grid_world: durumlar ve hareketlerin olduğu bir matris (5x5)
    gamma: discount factor (indirim faktörü)
    
    """
    
    for row in range(grid_world.shape[0]):
        for col in range(grid_world.shape[1]):
            curr_value=0

            for act in actions:
                reward, next_state= get_reward((row,col),act)

                #Sonraki state'e git
                next_state = grid_world[next_state]

                # Bellman equation : 1/4= 1/|A| (hareket sayısı)
                curr_value += (1/4)*(reward + gamma*next_state)

            # Her bir durumun değerini güncelle
            grid_world[row,col] = curr_value


#Haritayı Görselleştirme
            
def plot_grid(grid_world:np.ndarray,iterations:int=0):

    annot_kwargs = {"fontsize":18, "fontweight":"bold"}

    plt.figure(figsize=(10,10),dpi=100) #dpi: dots per inch(her bir inçteki nokta sayısı)


    # Haritayı görselleştir
    sns.heatmap(
        grid_world, annot=True, fmt=".2f", cmap="crest",
        annot_kws=annot_kwargs, cbar=False
    )

    #kenarlarda ki çizgileri kaldır
    plt.tick_params(
        bottom=False,
        labelbottom=False,
        left=False,
        labelleft=False
    )

    plt.title(f"Value Function (iteration={iterations})", fontsize=20, fontweight="bold")

    #Görseli kaydet
    plt.savefig(f"iteration_{iterations}.png")


if __name__ == "__main__":
    grid_world = np.zeros((5,5))
    
    actions = [act.value for act in ACTIONS]

    iter_to_save = {0,2,8,16,31}
    for i in range(40):

        prev_grid = np.copy(grid_world)

        value_function(grid_world)


        if i in iter_to_save:
            plot_grid(grid_world, i)

        #Eğer o anki durum, bir önceki duruma yakınlık toleransı içindeyse iterasyonu durdur.
        if np.allclose(prev_grid, grid_world, rtol=0.01):
            print('done', i)
            plot_grid(grid_world, i)
            break