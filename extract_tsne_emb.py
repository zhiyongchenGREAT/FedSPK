# %%
# set gpu
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "7"

# %%
# use wandb artifact outside a run

import wandb

api = wandb.Api()
artifact = api.artifact('greatml/FedSPK/Artifact0.8790510733321198:v0')

datadir = artifact.download()

# %%
def loadParameters(model, path, map_location="cuda:0"):

    self_state = model.state_dict()
    loaded_state = torch.load(path, map_location=map_location)

    for name, param in loaded_state.items():
        if '__L__.W' in name:
            continue
        
        origname = name
        if name not in self_state:
            name = name.replace("module.", "")
            if name not in self_state:
                name = "__S__."+name
                if name not in self_state:
                    print("#%s is not in the model."%origname)
                    continue

        if self_state[name].size() != loaded_state[origname].size():
            print("#Wrong parameter length: %s, model: %s, loaded: %s"%(origname, self_state[name].size(), loaded_state['model'][origname].size()))
            continue

        self_state[name].copy_(param)



# %%
# load the pytorch model

import torch

loaded_state = torch.load(datadir + '/STD: std_FL test compact testing, canoical.pth', map_location='cpu')

# %%
from src.SpeakerNet import SpeakerNet

model = SpeakerNet(model='X_vector', trainfunc='softmax', nPerSpeaker=1, Syncbatch=False, n_mels=40, nOut=192, spec_aug=False, nClasses=5994, additional_model=[True, False])


# %%
loadParameters(model, datadir + '/STD: std_FL test compact testing, canoical.pth')

# %%
import time
from src.DatasetLoader import loadWAV
import os, sys

def evaluateFromList(model, listfilename, print_interval=100, test_path='', num_eval=10, eval_frames=0, verbose=True):
    
    model = model.cuda()

    model.eval()
    
    labels       = []
    files       = []
    feats       = []
    tstart      = time.time()

    ## Read all lines
    with open(listfilename) as listfile:
        while True:
            line = listfile.readline()
            if (not line):
                break

            data = line.split()

            files.append(data[1])
            labels.append(data[0])

    f_labels = []
    ## Save all features to file
    for idx, (file, label) in enumerate(zip(files, labels)):

        inp1 = torch.FloatTensor(loadWAV(os.path.join(test_path,file), eval_frames, evalmode=True, num_eval=num_eval)).cuda()

        try:
            ref_feat = model.forward(inp1).detach().cpu()
        except:
            print("Error in file: ", file)
            continue

        feats.append(ref_feat)
        f_labels.append(label)

        telapsed = time.time() - tstart

        if (idx % print_interval == 0) and verbose:
            sys.stdout.write("\rReading %d of %d: %.2f Hz, embedding size %d"%(idx, len(files), idx/telapsed, ref_feat.size()[1]))

    tstart = time.time()

    return feats, f_labels

# %%
import pickle

for i in range(12):
    feats, labels = evaluateFromList(model, "../flearn_data/G%d_ID/train_list_G%d_ID.txt"%(i+1,i+1))

    # save python objects feats and labels
    with open('../FL_log/feats_G%d_ID.pkl'%i, 'wb') as f:
        pickle.dump(feats, f)
    with open('../FL_log/labels_G%d_ID.pkl'%i, 'wb') as f:
        pickle.dump(labels, f)


# Second (Centerized)
# %%
api = wandb.Api()
artifact = api.artifact('greatml/FedSPK/Artifact0.9596814134430157:v0')

datadir = artifact.download()

# %%
# load the pytorch model

import torch

loaded_state = torch.load(datadir + '/STD: centerized training.pth', map_location='cpu')

# %%
from src.SpeakerNet import SpeakerNet

model = SpeakerNet(model='X_vector', trainfunc='softmax', nPerSpeaker=1, Syncbatch=False, n_mels=40, nOut=192, spec_aug=False, nClasses=5994, additional_model=[True, False])


# %%
loadParameters(model, datadir + '/STD: centerized training.pth')

# %%
import pickle

for i in range(12):
    feats, labels = evaluateFromList(model, "../flearn_data/G%d_ID/train_list_G%d_ID.txt"%(i+1,i+1))

    # save python objects feats and labels
    with open('../FL_log/feats_G%d_ID_Cent.pkl'%i, 'wb') as f:
        pickle.dump(feats, f)
    with open('../FL_log/labels_G%d_ID_Cent.pkl'%i, 'wb') as f:
        pickle.dump(labels, f)
