import os
import time
import datetime
import pickle
import yaml
import threading
import logging

import wandb

import torch.nn as nn
from torch.utils.tensorboard import SummaryWriter

from src.server import Server
from src.utils import launch_tensor_board


if __name__ == "__main__":
    # GPU setting
    
    # read configuration file
    with open('./config.yaml') as c:
        configs = list(yaml.load_all(c, Loader=yaml.FullLoader))
    global_config = configs[0]["global_config"]
    data_config = configs[1]["data_config"]
    fed_config = configs[2]["fed_config"]
    optim_config = configs[3]["optim_config"]
    init_config = configs[4]["init_config"]
    model_config = configs[5]["model_config"]
    log_config = configs[6]["log_config"]
    eval_config = configs[7]["eval_config"]

    os.environ["CUDA_VISIBLE_DEVICES"] = global_config["use_GPU"]

    # modify dataset path for FL
    if fed_config["centerized_training"]:
        print('centeralized training')
        data_config["multi_dataset_files"] = [os.path.join(data_config["fl_dataset_path"], "train_list_G1-12.txt")]
        data_config["multi_train_paths"] = ['']
    else:
        data_config["multi_dataset_files"] = [os.path.join(data_config["fl_dataset_path"],
            "train_list_G%d.txt"%(i+1)) for i in range(fed_config["K"])]
        data_config["multi_train_paths"] = [''] * fed_config["K"]

    # setup group evaluation
    eval_config["group_listfilenames"] = [os.path.join(data_config["fl_dataset_path"], "ver_list_G%d_VER.txt"%(i+1)) for i in range(eval_config["group_size"]//2)] + \
    [os.path.join(data_config["fl_dataset_path"], "ver_list_G%d_VER_simp.txt"%(i+1)) for i in range(eval_config["group_size"]//2, eval_config["group_size"])] 

    # modify log_path to contain current time
    log_config["log_path"] = os.path.join(log_config["log_path"], str(datetime.datetime.now().strftime("%Y-%m-%d_%H:%M:%S")))

    # initiate TensorBaord for tracking losses and metrics
    writer = SummaryWriter(log_dir=log_config["log_path"], filename_suffix="FL")
    tb_thread = threading.Thread(
        target=launch_tensor_board,
        args=([log_config["log_path"], log_config["tb_port"], log_config["tb_host"]])
        ).start()
    time.sleep(3.0)

    # set the configuration of global logger
    logger = logging.getLogger(__name__)
    logging.basicConfig(
        filename=os.path.join(log_config["log_path"], log_config["log_name"]),
        level=logging.INFO,
        format="[%(levelname)s](%(asctime)s) %(message)s",
        datefmt="%Y/%m/%d/ %I:%M:%S %p")
    
    # display and log experiment configuration
    message = "\n[WELCOME] Unfolding configurations...!"
    print(message); logging.info(message)

    for config in configs:
        print(config); logging.info(config)
    print()

    # stack configs into a single dictionary
    configs_wandb = {**global_config, **data_config, **fed_config, **optim_config, **init_config, **model_config, **log_config, **eval_config}

    wandb.init(
        # set the wandb project where this run will be logged
        project="FedSPK",
        notes=global_config["notes"],
        tags=global_config["tags"],        
        # track hyperparameters and run metadata
        config=configs_wandb,
        dir='/nvme/zhiyong/wandb'
    )

    # initialize federated learning 
    central_server = Server(model_config, global_config, data_config, init_config, fed_config, optim_config, eval_config)
    
    if global_config["evaluate"]:
        central_server.evaluate_global_model(task=eval_config["eval_task"], model_path=eval_config["model_path"], listfilename=eval_config["listfilename"], testfile_path=eval_config["testfile_path"])
        exit()
    
    # central_server.setup()
    central_server.setup_speaker_server()

    # do federated learning
    # central_server.fit()
    central_server.fit_speaker()

    # save resulting losses and metrics
    # with open(os.path.join(log_config["log_path"], "result.pkl"), "wb") as f:
    #     pickle.dump(central_server.results, f)
    
    # bye!
    message = "...done all learning process!\n...exit program!"
    print(message); logging.info(message)
    time.sleep(3); exit()
