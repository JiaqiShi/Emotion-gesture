"""@wubowen"""

import sys
import os
import pickle
import numpy as np
import torch
from net.models import AudioOnly
from utils.dataprocess import shorten

DATA_PATH = r'/home/wu/mounts/Emo-gesture/test_set.pkl'
SAVE_PATH = r'evaluate/result'


if __name__ == "__main__":

    try:
        model_name = sys.argv[1]
    except IndexError:
        raise IndexError("Must provide model name as 2nd argument")
    else:
        cuda = torch.cuda.is_available()
        device = torch.device("cuda" if cuda else "cpu")
        print("Device: ", device)

        audio_data, motion_data, _ = pickle.load(open(DATA_PATH, 'rb'))
        audio_data, motion_data = shorten(audio_data, motion_data)

        net = AudioOnly(hidden_dim=256, input_dim=20, output_dim=30,
                        num_stack=2, return_sequences=True, return_states=False).to(device)
        net.load_state_dict(torch.load('saved_model/chkpt.pt'))
        net.eval()

        errors = []
        outputs = []

        for i, (input, label) in enumerate(zip(audio_data, motion_data)):
            input = torch.tensor(input).unsqueeze(0).to(device)
            output = net(input).view(-1, 10, 3).detach().cpu().numpy()
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
