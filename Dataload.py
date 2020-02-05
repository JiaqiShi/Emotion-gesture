#%%
import pandas as pd
import os
import numpy as np
#%%
class Dataloader():
    def __init__(self, csv_path, one_only = True):
        assert os.path.exists(csv_path)
        self.df = pd.read_csv(csv_path, index_col=0)
        self.df.sort_index(axis=0,ascending=True,inplace=True)
        self.one_only = one_only

    def get_audio_data(self, audio_feature_path, padding = False):
        assert os.path.exists(audio_feature_path)
        if self.one_only:
            df = self.df[self.df['gender'] == [s[0] for s in  self.df['sentence']]]
        else:
            df = self.df
        audio_data = []
        for index, row in df.iterrows():
            data_path = os.path.join(audio_feature_path, 'Session'+str(row['Session'])+'/'+row['dialog']+'/'+index+'.npy')
            with open(data_path,'rb') as f:
                data = np.load(f)
            audio_data.append(data)
        if padding:
            pass
        return audio_data

    def get_motion_data(self, motion_path, padding = False):
        assert os.path.exists(motion_path)
        if self.one_only:
            df = self.df[self.df['gender'] == [s[0] for s in  self.df['sentence']]]
        else:
            df = self.df
        motion_data = []
        for index, row in df.iterrows():
            data_path = os.path.join(motion_path, 'Session'+str(row['Session'])+'/'+row['dialog']+'.npy')
            with open(data_path,'rb') as f:
                data = np.load(f)
            motion_data.append(data[row['frame_start']:row['frame_end']])
        if padding:
            pass
        return motion_data

    def get_emo_label(self):
        if self.one_only:
            df = self.df[self.df['gender'] == [s[0] for s in  self.df['sentence']]]
        else:
            df = self.df
        emo_map = {'neu':0,'fru':1,'ang':2,'sad':3,'exc':4,'sur':5,'hap':6,'fea':7,'dis':8,'oth':9,'xxx':-1}
        return [emo_map[emo] for emo in df['emo']]
        # return list(emo_map[df['emo']])
#%%
if __name__ == '__main__':
    csv_path = r'/home/shi/emo_dataset/iemocap_utils/set_frame.csv'
    audio_feature_path = r'/home/shi/emo_dataset/Emo-gesture/Audio_features/MFCC12_0_D_A'
    motion_path = r'/home/shi/emo_dataset/iemocap_utils/3Dpoint_up_filted'
    dataload = Dataloader(csv_path,one_only=True)

    audio_data = dataload.get_audio_data(audio_feature_path)

    motion_data = dataload.get_motion_data(motion_path)

    emo_label = dataload.get_emo_label()

# %%