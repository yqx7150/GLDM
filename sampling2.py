# coding=utf-8
# Copyright 2020 The Google Research Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

# pylint: skip-file
# pytype: skip-file
"""Various sampling methods."""
import functools

import torch
import numpy as np
import abc

from models.utils import from_flattened_numpy, to_flattened_numpy, get_score_fn
from scipy import integrate
import sde_lib
from models import utils as mutils
from skimage.measure import compare_psnr,compare_ssim
import cv2
import os.path as osp
import matplotlib.pyplot as plt
import scipy.io as io

def write_kdata(Kdata,name):
    temp = np.log(1+abs(Kdata))    
    plt.axis('off')
    plt.imshow(abs(temp),cmap='gray')
    plt.savefig(osp.join('./result/',name),transparent=True, dpi=128, pad_inches = 0,bbox_inches = 'tight')

def write_Data(model_num,psnr,ssim):
    filedir="result.txt"
    with open(osp.join('./result/',filedir),"w+") as f:#a+
        f.writelines(str(model_num)+' '+'['+str(round(psnr, 2))+' '+str(round(ssim, 4))+']')
        f.write('\n')

def write_images(x,image_save_path):
    x = np.clip(x * 255, 0, 255).astype(np.uint8)
    cv2.imwrite(image_save_path, x)
  
def k2wgt(X,W):
    Y = np.multiply(X,W) 
    return Y

def wgt2k(X,W,DC):
    Y = np.multiply(X,1./W)
    Y[W==0] = DC[W==0] 
    return Y
    
_CORRECTORS = {}
_PREDICTORS = {}


def register_predictor(cls=None, *, name=None):
  """A decorator for registering predictor classes."""

  def _register(cls):
    if name is None:
      local_name = cls.__name__
    else:
      local_name = name
    if local_name in _PREDICTORS:
      raise ValueError(f'Already registered model with name: {local_name}')
    _PREDICTORS[local_name] = cls
    return cls

  if cls is None:
    return _register
  else:
    return _register(cls)


def register_corrector(cls=None, *, name=None):
  """A decorator for registering corrector classes."""

  def _register(cls):
    if name is None:
      local_name = cls.__name__
    else:
      local_name = name
    if local_name in _CORRECTORS:
      raise ValueError(f'Already registered model with name: {local_name}')
    _CORRECTORS[local_name] = cls
    return cls

  if cls is None:
    return _register
  else:
    return _register(cls)


def get_predictor(name):
  return _PREDICTORS[name]


def get_corrector(name):
  return _CORRECTORS[name]


def get_sampling_fn(config, sde, shape, inverse_scaler, eps):
  """Create a sampling function.

  Args:
    config: A `ml_collections.ConfigDict` object that contains all configuration information.
    sde: A `sde_lib.SDE` object that represents the forward SDE.
    shape: A sequence of integers representing the expected shape of a single sample.
    inverse_scaler: The inverse data normalizer function.
    eps: A `float` number. The reverse-time SDE is only integrated to `eps` for numerical stability.

  Returns:
    A function that takes random states and a replicated training state and outputs samples with the
      trailing dimensions matching `shape`.
  """

  sampler_name = config.sampling.method
  # Probability flow ODE sampling with black-box ODE solvers
  if sampler_name.lower() == 'ode':
    sampling_fn = get_ode_sampler(sde=sde,
                                  shape=shape,
                                  inverse_scaler=inverse_scaler,
                                  denoise=config.sampling.noise_removal,
                                  eps=eps,
                                  device=config.device)
  # Predictor-Corrector sampling. Predictor-only and Corrector-only samplers are special cases.
  elif sampler_name.lower() == 'pc':
    predictor = get_predictor(config.sampling.predictor.lower())
    corrector = get_corrector(config.sampling.corrector.lower())
    sampling_fn = get_pc_sampler(sde=sde,
                                 shape=shape,
                                 predictor=predictor,
                                 corrector=corrector,
                                 inverse_scaler=inverse_scaler,
                                 snr=config.sampling.snr,
                                 n_steps=config.sampling.n_steps_each,
                                 probability_flow=config.sampling.probability_flow,
                                 continuous=config.training.continuous,
                                 denoise=config.sampling.noise_removal,
                                 eps=eps,
                                 device=config.device)
  else:
    raise ValueError(f"Sampler name {sampler_name} unknown.")

  return sampling_fn


class Predictor(abc.ABC):
  """The abstract class for a predictor algorithm."""

  def __init__(self, sde, score_fn, probability_flow=False):
    super().__init__()
    self.sde = sde
    # Compute the reverse SDE/ODE
    self.rsde = sde.reverse(score_fn, probability_flow)
    self.score_fn = score_fn

  @abc.abstractmethod
  def update_fn(self, x, t):
    """One update of the predictor.

    Args:
      x: A PyTorch tensor representing the current state
      t: A Pytorch tensor representing the current time step.

    Returns:
      x: A PyTorch tensor of the next state.
      x_mean: A PyTorch tensor. The next state without random noise. Useful for denoising.
    """
    pass


class Corrector(abc.ABC):
  """The abstract class for a corrector algorithm."""

  def __init__(self, sde, score_fn, snr, n_steps):
    super().__init__()
    self.sde = sde
    self.score_fn = score_fn
    self.snr = snr
    self.n_steps = n_steps

  @abc.abstractmethod
  def update_fn(self, x, t):
    """One update of the corrector.

    Args:
      x: A PyTorch tensor representing the current state
      t: A PyTorch tensor representing the current time step.

    Returns:
      x: A PyTorch tensor of the next state.
      x_mean: A PyTorch tensor. The next state without random noise. Useful for denoising.
    """
    pass


@register_predictor(name='euler_maruyama')
class EulerMaruyamaPredictor(Predictor):
  def __init__(self, sde, score_fn, probability_flow=False):
    super().__init__(sde, score_fn, probability_flow)

  def update_fn(self, x, t):
    dt = -1. / self.rsde.N
    z = torch.randn_like(x)
    drift, diffusion = self.rsde.sde(x, t)
    x_mean = x + drift * dt
    x = x_mean + diffusion[:, None, None, None] * np.sqrt(-dt) * z
    return x, x_mean


@register_predictor(name='reverse_diffusion')
class ReverseDiffusionPredictor(Predictor):
  def __init__(self, sde, score_fn, probability_flow=False):
    super().__init__(sde, score_fn, probability_flow)
  
  # Alogrithm 2
  def update_fn(self, x, t):

    f, G = self.rsde.discretize(x, t) # 3
    z = torch.randn_like(x) # 4
    x_mean = x - f # 3
    x = x_mean + G[:, None, None, None] * z # 5  
    
    return x, x_mean


@register_predictor(name='ancestral_sampling')
class AncestralSamplingPredictor(Predictor):
  """The ancestral sampling predictor. Currently only supports VE/VP SDEs."""

  def __init__(self, sde, score_fn, probability_flow=False):
    super().__init__(sde, score_fn, probability_flow)
    if not isinstance(sde, sde_lib.VPSDE) and not isinstance(sde, sde_lib.VESDE):
      raise NotImplementedError(f"SDE class {sde.__class__.__name__} not yet supported.")
    assert not probability_flow, "Probability flow not supported by ancestral sampling"

  def vesde_update_fn(self, x, t):
    sde = self.sde
    timestep = (t * (sde.N - 1) / sde.T).long()
    sigma = sde.discrete_sigmas[timestep]
    adjacent_sigma = torch.where(timestep == 0, torch.zeros_like(t), sde.discrete_sigmas.to(t.device)[timestep - 1])
    score = self.score_fn(x, t)
    x_mean = x + score * (sigma ** 2 - adjacent_sigma ** 2)[:, None, None, None]
    std = torch.sqrt((adjacent_sigma ** 2 * (sigma ** 2 - adjacent_sigma ** 2)) / (sigma ** 2))
    noise = torch.randn_like(x)
    x = x_mean + std[:, None, None, None] * noise
    return x, x_mean

  def vpsde_update_fn(self, x, t):
    sde = self.sde
    timestep = (t * (sde.N - 1) / sde.T).long()
    beta = sde.discrete_betas.to(t.device)[timestep]
    score = self.score_fn(x, t)
    x_mean = (x + beta[:, None, None, None] * score) / torch.sqrt(1. - beta)[:, None, None, None]
    noise = torch.randn_like(x)
    x = x_mean + torch.sqrt(beta)[:, None, None, None] * noise
    return x, x_mean

  def update_fn(self, x, t):
    if isinstance(self.sde, sde_lib.VESDE):
      return self.vesde_update_fn(x, t)
    elif isinstance(self.sde, sde_lib.VPSDE):
      return self.vpsde_update_fn(x, t)


@register_predictor(name='none')
class NonePredictor(Predictor):
  """An empty predictor that does nothing."""

  def __init__(self, sde, score_fn, probability_flow=False):
    pass

  def update_fn(self, x, t):
    return x, x


@register_corrector(name='langevin')
class LangevinCorrector(Corrector):
  def __init__(self, sde, score_fn, snr, n_steps):
    super().__init__(sde, score_fn, snr, n_steps)
    if not isinstance(sde, sde_lib.VPSDE) \
        and not isinstance(sde, sde_lib.VESDE) \
        and not isinstance(sde, sde_lib.subVPSDE):
      raise NotImplementedError(f"SDE class {sde.__class__.__name__} not yet supported.")

  def update_fn(self, x1,x2,x3,x_mean,t):
    sde = self.sde
    score_fn = self.score_fn
    n_steps = self.n_steps
    target_snr = self.snr
    if isinstance(sde, sde_lib.VPSDE) or isinstance(sde, sde_lib.subVPSDE):
      timestep = (t * (sde.N - 1) / sde.T).long()
      alpha = sde.alphas.to(t.device)[timestep]
    else:
      alpha = torch.ones_like(t)
    
    # Algorithm 4
    for i in range(n_steps):
   
      grad1 = score_fn(x1, t) # 5 
      grad2 = score_fn(x2, t) # 5 
      grad3 = score_fn(x3, t) # 5 
      
      noise1 = torch.randn_like(x1) # 4 
      noise2 = torch.randn_like(x2) # 4
      noise3 = torch.randn_like(x3) # 4
      
      grad_norm1 = torch.norm(grad1.reshape(grad1.shape[0], -1), dim=-1).mean()
      noise_norm1 = torch.norm(noise1.reshape(noise1.shape[0], -1), dim=-1).mean()
      grad_norm2 = torch.norm(grad2.reshape(grad2.shape[0], -1), dim=-1).mean()
      noise_norm2 = torch.norm(noise2.reshape(noise2.shape[0], -1), dim=-1).mean()      
      grad_norm3 = torch.norm(grad3.reshape(grad3.shape[0], -1), dim=-1).mean()
      noise_norm3 = torch.norm(noise3.reshape(noise3.shape[0], -1), dim=-1).mean()            
      
      grad_norm =(grad_norm1+grad_norm2+grad_norm3)/3.0
      noise_norm = (noise_norm1+noise_norm2+noise_norm3)/3.0
      
      step_size =  (2 * alpha)*((target_snr * noise_norm / grad_norm) ** 2 ) # 6 
   
      x_mean = x_mean + step_size[:, None, None, None] * (grad1+grad2+grad3)/3.0 # 7
      
      x1 = x_mean + torch.sqrt(step_size * 2)[:, None, None, None] * noise1 # 7
      x2 = x_mean + torch.sqrt(step_size * 2)[:, None, None, None] * noise2 # 7
      x3 = x_mean + torch.sqrt(step_size * 2)[:, None, None, None] * noise3 # 7
      
    return x1,x2,x3,x_mean


@register_corrector(name='ald')
class AnnealedLangevinDynamics(Corrector):
  """The original annealed Langevin dynamics predictor in NCSN/NCSNv2.

  We include this corrector only for completeness. It was not directly used in our paper.
  """

  def __init__(self, sde, score_fn, snr, n_steps):
    super().__init__(sde, score_fn, snr, n_steps)
    if not isinstance(sde, sde_lib.VPSDE) \
        and not isinstance(sde, sde_lib.VESDE) \
        and not isinstance(sde, sde_lib.subVPSDE):
      raise NotImplementedError(f"SDE class {sde.__class__.__name__} not yet supported.")

  def update_fn(self, x, t):
    sde = self.sde
    score_fn = self.score_fn
    n_steps = self.n_steps
    target_snr = self.snr
    if isinstance(sde, sde_lib.VPSDE) or isinstance(sde, sde_lib.subVPSDE):
      timestep = (t * (sde.N - 1) / sde.T).long()
      alpha = sde.alphas.to(t.device)[timestep]
    else:
      alpha = torch.ones_like(t)

    std = self.sde.marginal_prob(x, t)[1]
   
    for i in range(n_steps):
      grad = score_fn(x, t)
      noise = torch.randn_like(x)
      step_size = (target_snr * std) ** 2 * 2 * alpha
      x_mean = x + step_size[:, None, None, None] * grad
      x = x_mean + noise * torch.sqrt(step_size * 2)[:, None, None, None]
    return x, x_mean


@register_corrector(name='none')
class NoneCorrector(Corrector):
  """An empty corrector that does nothing."""

  def __init__(self, sde, score_fn, snr, n_steps):
    pass

  def update_fn(self, x, t):
    return x, x


def shared_predictor_update_fn(x, t, sde, model, predictor, probability_flow, continuous):
  """A wrapper that configures and returns the update function of predictors."""
  score_fn = mutils.get_score_fn(sde, model, train=False, continuous=continuous)
  if predictor is None:
    # Corrector-only sampler
    predictor_obj = NonePredictor(sde, score_fn, probability_flow)
  else:
    predictor_obj = predictor(sde, score_fn, probability_flow)
  return predictor_obj.update_fn(x, t)


def shared_corrector_update_fn(x1,x2,x3,x_mean,t,sde, model, corrector, continuous, snr, n_steps):
  """A wrapper tha configures and returns the update function of correctors."""
  score_fn = mutils.get_score_fn(sde, model, train=False, continuous=continuous)
  if corrector is None:
    # Predictor-only sampler
    corrector_obj = NoneCorrector(sde, score_fn, snr, n_steps)
  else:
    corrector_obj = corrector(sde, score_fn, snr, n_steps)
  return corrector_obj.update_fn(x1,x2,x3,x_mean,t)


def get_pc_sampler(sde, shape, predictor, corrector, inverse_scaler, snr,
                   n_steps=1, probability_flow=False, continuous=False,
                   denoise=True, eps=1e-3, device='cuda'):
  # Create predictor & corrector update functions
  predictor_update_fn = functools.partial(shared_predictor_update_fn,
                                          sde=sde,
                                          predictor=predictor,
                                          probability_flow=probability_flow,
                                          continuous=continuous)
  corrector_update_fn = functools.partial(shared_corrector_update_fn,
                                          sde=sde,
                                          corrector=corrector,
                                          continuous=continuous,
                                          snr=snr,
                                          n_steps=n_steps)

  def pc_sampler(model):
    """ The PC sampler funciton.
    Args:
      model: A score model.
    Returns:
      Samples, number of function evaluations.
    """
    with torch.no_grad():
    
      # Initial sample
      #x = sde.prior_sampling(shape).to(device) # 1
      timesteps = torch.linspace(sde.T, eps, sde.N, device=device)
      
      file_path='/home/lqg/桌面/ncsn++/input_data/SIAT_test_image31/test_data_0'+str(5)+'.mat'
      ori_data = io.loadmat(file_path)['Img']
      ori_data = ori_data/np.max(abs(ori_data))
      write_images(abs(ori_data),osp.join('./result/'+'ori'+'.png'))
      
      #======== mask
      mask = io.loadmat('/home/lqg/桌面/ncsn++/input_data/mask/mask_radial_030.mat')['mask_radial_030']  
      #mask = io.loadmat('/home/lqg/桌面/TZJ_SDE/input_data/mask/random_mask_r3_256.mat')['mask']  
      #mask = io.loadmat('./input_data/mask/poisson/2.mat')['mask'] 
      #mask = io.loadmat('/home/lqg/桌面/TZJ_SDE/input_data/mask/random2D/2.mat')['mask']
      #mask = np.fft.fftshift(io.loadmat('/home/lqg/桌面/TZJ_SDE/input_data/mask/mask_cart85.mat')['mask_cart85'])        
      #temp = np.random.uniform(0,1,size=(256,256))
      #mask = (temp>=0.9)+0 
      #mask[127-25:127+25,127-25:127+25] = 1
      print(np.sum(mask)/65536)
      #plt.imshow(abs(mask),cmap='gray')
      #plt.show()
      write_images(abs(mask),osp.join('./result/'+'mask'+'.png'))
      #========      
      weight = io.loadmat('/home/lqg/桌面/ncsn++/input_data/weight1.mat')['weight']    
    
      Kdata=np.fft.fftshift(np.fft.fft2(ori_data))
      ori_data = np.fft.ifft2(Kdata)
    
      Ksample=np.multiply(mask,Kdata)
      Zeorfilled_data=np.fft.ifft2(Ksample)   
      k_w = k2wgt(Ksample,weight)
      
      #write_kdata(Kdata,'kdata')
      #write_kdata(Ksample,'ksample')
    
      psnr_zero=compare_psnr(255*abs(Zeorfilled_data),255*abs(ori_data),data_range=255)
      ssim_zero=compare_ssim(abs(Zeorfilled_data),abs(ori_data),data_range=1)
      print('psnr_zero: ',psnr_zero,'ssim_zero: ',ssim_zero)
      write_images(abs(Zeorfilled_data),osp.join('./result/'+'Zeorfilled_'+str(round(psnr_zero, 2))+str(round(ssim_zero, 4))+'.png'))
      
      x_input = np.random.uniform(-1,1,size=(1,6,256,256))
      #x_input[0,0,:,:] = np.real(k_w)
      #x_input[0,1,:,:] = np.imag(k_w)
      #x_input[0,2,:,:] = np.real(k_w)
      #x_input[0,3,:,:] = np.imag(k_w)
      #x_input[0,4,:,:] = np.real(k_w)
      #x_input[0,5,:,:] = np.imag(k_w)
      x_input = torch.from_numpy(x_input).to(device)
      x_mean = torch.tensor(x_input,dtype=torch.float32).cuda()
      x1 = x_mean
      x2 = x_mean
      x3 = x_mean 
              
      max_psnr = 0
      max_ssim = 0

      for i in range(sde.N):
        print('======== ',i)
        t = timesteps[i]
        vec_t = torch.ones(shape[0], device=t.device) * t        
        ##======================================================= Predictor
        
        x, x_mean = predictor_update_fn(x_mean, vec_t, model=model)       
        x_mean = x_mean.cpu().numpy() # (1,6,256,256)    
        x_mean = np.array(x_mean,dtype=np.float32) 
             
        kw_real = (x_mean[0,0,:,:]+x_mean[0,2,:,:]+x_mean[0,4,:,:])/3
        kw_imag = (x_mean[0,1,:,:]+x_mean[0,3,:,:]+x_mean[0,5,:,:])/3
        k_w = kw_real+1j*kw_imag
                
        # DC 1        
        k_complex = wgt2k(k_w,weight,Ksample)
        k_complex2 = Ksample + k_complex*(1-mask)
        
        # BACK 1
        k_w = k2wgt(k_complex2,weight)
        x_input = np.zeros((1,6,256,256),dtype=np.float32)
        x_input[0,0,:,:] = np.real(k_w)
        x_input[0,1,:,:] = np.imag(k_w)
        x_input[0,2,:,:] = np.real(k_w)
        x_input[0,3,:,:] = np.imag(k_w)
        x_input[0,4,:,:] = np.real(k_w)
        x_input[0,5,:,:] = np.imag(k_w)
        x_mean = torch.tensor(x_input,dtype=torch.float32).cuda()

        
        ##======================================================= Corrector
        
        x1,x2,x3,x_mean = corrector_update_fn(x1,x2,x3,x_mean, vec_t, model=model) 
              
        x_mean = x_mean.cpu().numpy() # (1,3,256,256)  
        x_mean = np.array(x_mean,dtype=np.float32)
            
           
        kw_real = (x_mean[0,0,:,:]+x_mean[0,2,:,:]+x_mean[0,4,:,:])/3
        kw_imag = (x_mean[0,1,:,:]+x_mean[0,3,:,:]+x_mean[0,5,:,:])/3
        k_w = kw_real+1j*kw_imag
                
        # DC 2       
        k_complex = wgt2k(k_w,weight,Ksample)
        k_complex2 = Ksample + k_complex*(1-mask)
        rec_Image = np.fft.ifft2(k_complex2)
      
        # PSNR
        #rec_Image = rec_Image/np.max(np.abs(rec_Image))
        #rec_Image = np.clip(rec_Image,-1,1)
        psnr = compare_psnr(255*abs(rec_Image),255*abs(ori_data),data_range=255)
        ssim = compare_ssim(abs(rec_Image),abs(ori_data),data_range=1)
        print(' PSNR:', psnr,' SSIM:', ssim)  
      
        if max_ssim<=ssim:
          max_ssim = ssim
          
        if max_psnr<=psnr:
          max_psnr = psnr
          write_Data('checkpoint',max_psnr,ssim) 
          write_images(abs(rec_Image),osp.join('./result/'+'Rec'+'.png'))
          #krec = np.fft.fft2(rec_Image)
          #write_kdata(krec,'k_Rec')
        
        # BACK 2
        k_w = k2wgt(k_complex2,weight)
        x_input = np.zeros((1,6,256,256),dtype=np.float32)
        x_input[0,0,:,:] = np.real(k_w)
        x_input[0,1,:,:] = np.imag(k_w)
        x_input[0,2,:,:] = np.real(k_w)
        x_input[0,3,:,:] = np.imag(k_w)
        x_input[0,4,:,:] = np.real(k_w)
        x_input[0,5,:,:] = np.imag(k_w)
        x_mean = torch.tensor(x_input,dtype=torch.float32).cuda() 
            
        x_mean = x_mean.to(device)          
    
        
        
      return x_mean #inverse_scaler(x_mean if denoise else x), sde.N * (n_steps + 1)

  return pc_sampler


def get_ode_sampler(sde, shape, inverse_scaler,
                    denoise=False, rtol=1e-5, atol=1e-5,
                    method='RK45', eps=1e-3, device='cuda'):
  """Probability flow ODE sampler with the black-box ODE solver.

  Args:
    sde: An `sde_lib.SDE` object that represents the forward SDE.
    shape: A sequence of integers. The expected shape of a single sample.
    inverse_scaler: The inverse data normalizer.
    denoise: If `True`, add one-step denoising to final samples.
    rtol: A `float` number. The relative tolerance level of the ODE solver.
    atol: A `float` number. The absolute tolerance level of the ODE solver.
    method: A `str`. The algorithm used for the black-box ODE solver.
      See the documentation of `scipy.integrate.solve_ivp`.
    eps: A `float` number. The reverse-time SDE/ODE will be integrated to `eps` for numerical stability.
    device: PyTorch device.

  Returns:
    A sampling function that returns samples and the number of function evaluations during sampling.
  """

  def denoise_update_fn(model, x):
    score_fn = get_score_fn(sde, model, train=False, continuous=True)
    # Reverse diffusion predictor for denoising
    predictor_obj = ReverseDiffusionPredictor(sde, score_fn, probability_flow=False)
    vec_eps = torch.ones(x.shape[0], device=x.device) * eps
    _, x = predictor_obj.update_fn(x, vec_eps)
    return x

  def drift_fn(model, x, t):
    """Get the drift function of the reverse-time SDE."""
    score_fn = get_score_fn(sde, model, train=False, continuous=True)
    rsde = sde.reverse(score_fn, probability_flow=True)
    return rsde.sde(x, t)[0]

  def ode_sampler(model, z=None):
    """The probability flow ODE sampler with black-box ODE solver.

    Args:
      model: A score model.
      z: If present, generate samples from latent code `z`.
    Returns:
      samples, number of function evaluations.
    """
    with torch.no_grad():
      # Initial sample
      if z is None:
        # If not represent, sample the latent code from the prior distibution of the SDE.
        x = sde.prior_sampling(shape).to(device)
      else:
        x = z

      def ode_func(t, x):
        x = from_flattened_numpy(x, shape).to(device).type(torch.float32)
        vec_t = torch.ones(shape[0], device=x.device) * t
        drift = drift_fn(model, x, vec_t)
        return to_flattened_numpy(drift)

      # Black-box ODE solver for the probability flow ODE
      solution = integrate.solve_ivp(ode_func, (sde.T, eps), to_flattened_numpy(x),
                                     rtol=rtol, atol=atol, method=method)
      nfe = solution.nfev
      x = torch.tensor(solution.y[:, -1]).reshape(shape).to(device).type(torch.float32)

      # Denoising is equivalent to running one predictor step without adding noise
      if denoise:
        x = denoise_update_fn(model, x)

      x = inverse_scaler(x)
      return x, nfe

  return ode_sampler
