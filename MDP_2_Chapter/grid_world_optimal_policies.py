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


# get a reward based on the state and action
def get_reward(state:tuple, act:tuple):
    """
    state: o anki durum (x(satır),y(sütun))

    act : hareket (d_row(delta-row),d_col(delta-col)) aralarındaki farklar

    output: reward and next state (ödülü ve bir sonraki durumu döndürür)
    """
    if state == (0, 1):
        return 10, (4, 1)
    if state == (0, 3):
        return 5, (2, 3)
    
    #Hareketi güncelleme
    next_row = state[0] + act[0]
    next_col = state[1] + act[1]

    if not (0 <= next_row < 5 and 0 <= next_col < 5):
        return -1, state

    return 0, (next_row, next_col)



# value function
def value_update(grid_world:np.ndarray, 
                 actions:list,gamma=0.9):
    """
    grid_world: durumlar ve hareketlerin olduğu bir matris (5x5)
    gamma: discount factor (indirim faktörü)
    """


    # Hex code for (right, down, left, up)
    act_lists = np.array([0x2190, 0x2193, 0x2192, 0x2191])

    #Policy için acts
    optim_acts = []


    for row in range(5):

        #Tüm hareketler için acts
        acts = []


        for col in range(5):

            #Hareketlerin değerlerini sıfırla
            value_candidates = np.zeros(shape=(len(actions)))


            # Tüm hareket opsiyonları için döngü
            for i , act in enumerate(actions):

                reward, next_state = get_reward((row, col), act) # Ödül ve bir sonraki durumu al.


                next_value = grid_world[next_state] #Bir sonraki duruma geç.


                # İndirimli value function hesapla
                value_candidates[i] = reward + gamma * next_value
            
            #Tüm aksiyonlar arasında hesaplana en büyük değeri al ve ilgili kutuya yaz.
            grid_world[row, col] = value_candidates.max()

            # en büyük değere sahip aksiyonları al.
            max_args = np.where(value_candidates == value_candidates.max())
            
            #En büyük değere sahip aksiyonlarının hex code'larını al ve selected_acts'e ekle.
            selected_acts = ''.join([chr(c) for 
                                        c in act_lists[max_args].tolist()])
            acts.append(selected_acts)

        optim_acts.append(acts)

    return optim_acts

#Görselleştirme

def plot_grid(grid_world:np.ndarray, annot=None):
    annot_kwargs = {
        'fontsize': '18',
        # 'fontweight': 'bold'
    }
    fmt = ''
    file_name = 'optimal_policy_based_functions'
    
    if annot == None:
        annot = True
        fmt = '.1f'
        file_name = 'optimal_values_based_functions'
    
    plt.figure(figsize=(6, 6), dpi=100)
    sns.heatmap(grid_world, 
                annot=annot,
                annot_kws=annot_kwargs,
                linewidths=1., 
                # fmt='.1f',
                fmt=fmt,
                cmap='crest',
                cbar=False)
    plt.tick_params(
                bottom=False,
                labelbottom=False,
                left=False,
                labelleft=False)
    plt.title(f"{' '.join(file_name.split('_')).capitalize()}", fontsize=20, fontweight='bold', pad=10)
    plt.savefig(f'plots/example_3_8/{file_name}.png')



if __name__ == "__main__":
    
    # GridWorld'ü oluştur
    grid_world = np.zeros(shape=(5, 5))
    actions = [action.value for action in ACTIONS]

    for i in range(40):
        prev_grid = np.copy(grid_world)

        #[][][][] şeklinde tüm kutular için en iyi hareketleri verir.
        optim_acts = value_update(grid_world, actions)
        
        


        if np.allclose(prev_grid, grid_world, rtol=0.001):
            print('done', i)
            print(optim_acts)
            plot_grid(grid_world, annot = optim_acts) #optimal based functions için
            plot_grid(grid_world, annot = None) # value based functions için
            break