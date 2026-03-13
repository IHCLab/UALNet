from os.path import join

# path 
f_to_save = ["UALNet.py","PriorNet.py","module.py","dataloader.py","init.py","Train.py","config.py","D_net.py"]
save_path = "./UALNet_Traing"
save_result =  join(save_path, "result")
train_data_path = "./Train/"
test_data_path  = "./Test/"
valid_data_path = "./Valid/"

# generate data
batch_size  = 2
gpu_ids     = 0

#PriroNet
PriorNet_input_channel = 12
PriorNet_output_channel = 12
PriorNet_feat_dim = 12
PriorNet_SPM_dim = 186
PriorNet_num_stages = 3
PriorNet_lr = 5e-4
PriorNet_opt = 'Adam'
Prior_epoch_num = 600
PriorNet_test_period = 5

# Discriminator
Discriminator_in_channel = 186
Discriminator_hidden_channels = 12
Discriminator_lr = 1e-5
Discriminator_opt = 'Adam'

# UALNet
UALNet_lambda_1 = 5e-4
UALNet_lambda_2 = 0.5
UALNet_mu = 0.05
UALNet_input_channel = 12
UALNet_output_channel = 186
UALNet_num_iter = 2
UALNet_lr = 5e-4
UALNet_opt = 'Adam'
UALNet_period = 5
UALNet_epoch_num = 600
UALNet_test_period = 5



