"""@wubowen"""

import sys
import os
import pickle
import numpy as np
import torch
from net.models import AudioEmotion
from utils.dataprocess import shorten
from utils.helpers import one_hot

DATA_PATH = r'/home/wu/mounts/Emo-gesture/test_set.pkl'
SAVE_PATH = r'evaluate/result'


if __name__ == "__main__":

    if len(sys.argv) < 2:
        raise ValueError("Must provide model_name as 2nd argument")

    model_name = sys.argv[1]

    cuda = torch.cuda.is_available()
    device = torch.device("cuda" if cuda else "cpu")
    print("Device: ", device)

    audio_data, motion_data, emo_data = pickle.load(open(DATA_PATH, 'rb'))
    audio_data, motion_data = shorten(audio_data, motion_data)
    emo_data = list(map(one_hot, emo_data))

    net = AudioEmotion(hidden_dim=128, x_input_dim=20, e_input_dim=11, output_dim=30,
                       num_stack=2, return_sequences=True, return_states=False).to(device)
    net.load_state_dict(torch.load('saved_model/' + model_name + '.pt'))
    net.eval()

    errors = []
    outputs = []

    for i, (x_input, e_input, label) in enumerate(zip(audio_data, emo_data, motion_data)):
        x_input = torch.tensor(x_input).unsqueeze(0).to(device)
        e_input = torch.tensor(e_input, dtype=torch.float).unsqueeze(0).to(device)
        output = net(x_input, e_input).view(-1, 10, 3).detach().cpu().numpy()
        error = np.mean(np.abs(output, label))
        errors.append(error)
        print(output.shape)
        outputs.append(output)

    ave_error = np.mean(errors)
    print("Average errors:", ave_error)

    # save predictions
    try:
        dirname = os.path.join(SAVE_PATH, model_name)
        os.mkdir(dirname)
    except:
        print("Dir already exists")
    finally:
        with open(os.path.join(
                dirname, 'test_predicts.pkl'), 'wb+') as f:
            pickle.dump(outputs, f)
