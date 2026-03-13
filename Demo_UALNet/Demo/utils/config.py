from os.path import join

#Inference 
gpu_ids = 0
data_path = ["ang20190713t025848_47",
             "ang20191019t173905_4",
             "ang20191027t153848_7",
             "ang20191027t153111_10"]

best_priorNet_path = './Checkpoint/PriorNet_the_best.pth'
best_discriminator_path = './Checkpoint/Discriminator_the_best.pth'
best_ualnet_path = './Checkpoint/UALNet_the_best.pth'

#PriroNet
PriorNet_input_channel = 12
PriorNet_output_channel = 12
PriorNet_feat_dim = 12
PriorNet_SPM_dim = 186
PriorNet_num_stages = 3

# Discriminator
Discriminator_in_channel = 186
Discriminator_hidden_channels = 12

# UALNet
UALNet_lambda_1 = 5e-4
UALNet_lambda_2 = 0.5
UALNet_mu = 0.05
UALNet_input_channel = 12
UALNet_output_channel = 186
UALNet_num_iter = 2




