import torch
import time
import os
import numpy as np
import scipy.io as sio
import utils.config as cfg
import warnings
from utils.fun import *
from einops import rearrange
from arch.PriorNet import PriorNet 
from arch.D_net import Discriminator 
from arch.UALNet import UALNet 
warnings.filterwarnings("ignore")

device = ( torch.device(f"cuda:{0}")
            if torch.cuda.is_available()
            else torch.device("cpu") 
         )

def load_models():
    priorNet = PriorNet(
            in_channels=cfg.PriorNet_input_channel,
            out_channels=cfg.PriorNet_output_channel,
            feat_dim=cfg.PriorNet_feat_dim,
            hyperspectral_channels=cfg.PriorNet_SPM_dim,
            num_stages=cfg.PriorNet_num_stages,
        ).to(device)
    
    discriminator = Discriminator(
            in_channels=cfg.Discriminator_in_channel,
            hidden_channels=cfg.Discriminator_hidden_channels,
        ).to(device)
    
    ualNet = UALNet(
            lambda_1=cfg.UALNet_lambda_1,
            lambda_2=cfg.UALNet_lambda_2,
            mu=cfg.UALNet_mu,
            dim=cfg.UALNet_output_channel,
            s_channel=cfg.UALNet_input_channel,
            num_iter=cfg.UALNet_num_iter,
        ).to(device)
    
    # load model
    p_ckpt = torch.load(cfg.best_priorNet_path, map_location=device)
    ual_ckpt = torch.load(cfg.best_ualnet_path, map_location=device)
    d_ckpt = torch.load(cfg.best_discriminator_path, map_location=device)
    
    priorNet.load_state_dict(p_ckpt["model_state_dict"])
    discriminator.load_state_dict(d_ckpt["model_state_dict"])
    ualNet.load_state_dict(ual_ckpt["model_state_dict"])

    return priorNet,discriminator,ualNet
    

if __name__ == "__main__":
    with torch.no_grad():

        print('-' * 50)
        print("Start Converting Sentinel-2 MSI to AVIRIS HSI via UALNet: ")
        priorNet,discriminator,ualNet = load_models()
        priorNet.eval()
        discriminator.eval()
        ualNet.eval()

        for name in cfg.data_path:
            pth = os.path.join('./Demo_data',name)+".mat"
            data = sio.loadmat(pth)
            AVIRIS_ref = data['AVIRIS']
            Sentinel2  = data['Sentinel2']
            Sentinel2  = torch.from_numpy(Sentinel2.astype(np.float32)).permute(2,0,1).unsqueeze(0).to(device)

        
            # Model Inference
            t0 = time.time()
            s_u,p= priorNet(Sentinel2)
            with torch.enable_grad():
                 pred_hsi = ualNet(s_u, discriminator, p)
            t1 = time.time()
            
            # converting
            s_2 = Sentinel2.cpu().numpy()[0, ...].transpose(1,2,0)
            s_u = s_u.cpu().numpy()[0, ...].transpose(1,2,0)
            pred_hsi = pred_hsi.detach().cpu().numpy()[0, ...].transpose(1,2,0)
            
            idx_time = t1-t0
        
            print("time :", round(idx_time,4))
            save_dir = os.path.join('./Result',name)
          
            plot_reconstruction(AVIRIS_ref,
                s_2,
                pred_hsi,
                s_u,
                save_dir=save_dir,
                num_signature_plot=4,
                band_for_rgb_1based=(25, 12, 8),
                seed=0,
                )
                        
           