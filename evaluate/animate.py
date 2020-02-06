#%%
import os
import pickle
import numpy as np
from matplotlib import pyplot as plt
from matplotlib.animation import FuncAnimation


def animation(pre_motion, true_motion, save_path=None):
    fig, (ax1, ax2) = plt.subplots(1, 2)

    def update(i):
        lines = [[0, 1], [1, 2], [2, 3], [2, 4], [4, 5], [5, 6], [2, 7],
                 [7, 8], [8, 9]]
        label = 'timestep {}'.format(i)
        print(label)
        xs = [pre_motion[i, :, 0], true_motion[i, :, 0]]
        ys = [-pre_motion[i, :, 1], -true_motion[i, :, 1]]
        axs = [ax1, ax2]
        for ax, x, y in zip(axs, xs, ys):
            ax.cla()
            ax.axis('equal')
            ax.set_xlim(-0.5, 0.5)
            ax.set_ylim(-0.5, 0.5)
            ax.set_xlabel(label)
            ax.scatter(x, y)
            for line in lines:
                ax.plot(x[line], y[line])
        return fig, ax1, ax2

    anim = FuncAnimation(fig,
                         update,
                         frames=np.arange(pre_motion.shape[0]),
                         interval=50)
    anim.save(save_path, dpi=80, writer='imagemagick')


def sample_animate(result_path,
                   label_path,
                   *,
                   output_path='result/gif',
                   total_num=2):
    ''' 
        Sample some of generated motion and animate
        result_path: path of generated motion
        label_path: path of true motion
        output_path: gif file folder
        total_num = #sampled motion
    '''
    with (open(result_path, 'rb')) as f:
        result = pickle.load(f)
    with (open(true_path, 'rb')) as f:
        _, true_data, _ = pickle.load(f)
    if not os.path.exists(output_path):
        os.makedirs(output_path)
    sample_num = np.random.randint(0, len(result), total)
    result = [result[q] for q in sample_num]
    true_data = [true_data[q] for q in sample_num]
    for num, (pre_motion, true_motion) in enumerate(zip(result, true_data)):
        save_path = os.path.join(output_path, str(num) + '.gif')
        animation(pre_motion, true_motion, save_path)


if __name__ == '__main__':
    result_path = 'result/chkpt/test_predicts.pkl'
    true_path = '/home/shi/emo_dataset/Emo-gesture/test_set.pkl'
    sample_animate(result_path, true_path)

# %%
