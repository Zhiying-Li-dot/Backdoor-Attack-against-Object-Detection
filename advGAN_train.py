from time import sleep
import torch.nn as nn
import torch
import numpy as np
import torch.nn.functional as F
import torchvision
import os
from log import logger
from torchvision.ops import box_iou
from loss import *
from infected_model import *
from frequency_detector import *
from datasets import test_dataloader, test_length, device, invisible_test_dataloader, invisible_test_length
from tqdm import tqdm
import advGAN_model
import wandb
from torch_dct import dct_2d, idct_2d

from config import cfg
cfg = cfg['advGAN']

# custom weights initialization called on netG and netD
def weights_init(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1 and classname.find('Gaussian') == -1:
        nn.init.kaiming_normal_(m.weight, nonlinearity='relu')

class AdvGAN_Attack:
    def __init__(self,
                 device,
                 image_nc,
                 box_min,
                 box_max):
        output_nc = image_nc
        self.device = device
        self.input_nc = image_nc
        self.output_nc = output_nc
        self.box_min = box_min
        self.box_max = box_max
        self.target_criterion = torch.nn.BCEWithLogitsLoss()
        self.detector_criterion = torch.nn.BCEWithLogitsLoss()

        self.gen_input_nc = image_nc
        self.netG = advGAN_model.Generator(self.gen_input_nc, image_nc).to(device)
        self.netDisc = advGAN_model.Discriminator(image_nc).to(device)

        # initialize all weights
        # self.netG.apply(weights_init)
        # self.netDisc.apply(weights_init)

        # initialize optimizers
        self.optimizer_G = torch.optim.Adam(self.netG.parameters(),
                                            lr=cfg['G_lr1'])
        self.optimizer_D = torch.optim.Adam(self.netDisc.parameters(),
                                            lr=cfg['D_lr1'])
        wandb.watch(self.netG)
        self.visibility_loss = VisiblilityLoss(cfg['bound'])
        # self.visibility_loss = nn.MSELoss()
        self.shapley_loss = ShapleyLoss()
        self.decision_loss = DecisionLoss()
        self.poison_loss = PoisonLoss()
        self.frequency_loss = FrequencyLoss()
        self.gather_loss = GatherLoss()
        if not os.path.exists(cfg['models_path']):
            os.makedirs(cfg['models_path'])

    def train_batch(self, x, invisible_x):
        # optimize D
        f_ix = dct_2d(invisible_x, 'ortho')
        f_x = dct_2d(x, 'ortho')
        # condition = (f_ix - f_x)
        # condition = torch.where(torch.isnan(condition), torch.full_like(condition, 1e-6), condition)
        # condition = torch.where(torch.isinf(condition), torch.full_like(condition, 1e6), condition)
        condition = invisible_x - x
        perturbation = self.netG(invisible_x, condition)
        adv_images = torch.clamp(invisible_x + torch.clamp(perturbation, -0.3, 0.3), 0, 1)
        # adv_images = self.netG(invisible_x, condition)

        # add a clipping trick
        # clamp_min = float(cfg['clamp_min'])
        # clamp_max = float(cfg['clamp_max'])
        # perturbation = torch.clamp(perturbation, clamp_min, clamp_max)
        # f_adv_images = perturbation + f_x
        # adv_images = torch.clamp(idct_2d(f_adv_images, 'ortho'), 0, 1)
        # perturbation = adv_images - x
        # f_adv_images = dct_2d(adv_images, 'ortho')
        # perturbation = self.netG(x)
        # adv_images = torch.clamp(perturbation + torch.clamp(x, -0.1, 0.1), 0, 1)
        f_adv_images = dct_2d(adv_images.detach(), 'ortho')
        # f_x = dct_2d(x, 'ortho')

        # # add a clipping trick
        # clamp_min = float(cfg['clamp_min'])
        # clamp_max = float(cfg['clamp_max'])
        # adv_images = torch.clamp(perturbation, clamp_min, clamp_max) + x
        # adv_images = torch.clamp(adv_images, self.box_min, self.box_max)

        # self.optimizer_D.zero_grad()
        # pred_real = self.netDisc(x)
        # loss_D_real = F.mse_loss(pred_real, torch.ones_like(pred_real, device=self.device))
        # loss_D_real.backward(retain_graph=True)
        
        # pred_half_real = self.netDisc(dct_2d(invisible_x, 'ortho'))
        pred_half_real = self.netDisc(f_ix)
        loss_D_half_real = F.binary_cross_entropy(pred_half_real, torch.ones_like(pred_half_real, device=self.device))
        # loss_D_half_real.backward(retain_graph=True)

        pred_fake = self.netDisc(f_adv_images.detach())
        loss_D_fake = F.binary_cross_entropy(pred_fake, torch.zeros_like(pred_fake, device=self.device))
        # loss_D_fake.backward()
        
        loss_D_GAN = (loss_D_fake + pred_half_real).mean()
        loss_D_GAN.backward()
        self.optimizer_D.step()

        # optimize G
        self.optimizer_G.zero_grad()

        # cal G's loss in GAN
        
        pred_fake = self.netDisc(f_adv_images)
        loss_G_fake = F.mse_loss(pred_fake, torch.ones_like(pred_fake, device=self.device))
        loss_G_fake.backward(retain_graph=True)

        # calculate perturbation norm
        # hinge loss
        # C = float(cfg['hinge_c'])
        # loss_perturb = torch.mean(torch.norm(perturbation.view(perturbation.shape[0], -1), 2, dim=1))
        # loss_perturb = torch.max(loss_perturb - C, torch.zeros(1, device=self.device))
        # loss_perturb = torch.tensor(0., device=device)

        # cal detector loss
        from frequency_detector import detector
        loss_det = self.frequency_loss(detector(f_adv_images))

        infected_output = infected_predict(adv_images)
        clean_output = clean_predict(adv_images)
        # cal adv loss
        loss_adv = self.decision_loss(infected_output)
        loss_adv.requires_grad = True
        
        loss_poison = self.poison_loss(infected_output, clean_output)
        
        # cal gather loss
        # f_perturbation = f_adv_images - f_x
        loss_gather = self.gather_loss(adv_images - x)
        # loss_gather = torch.tensor(0)
        
        # cal visibility loss
        # new_perturbation = torch.full_like(perturbation, 0.3)
        # new_perturbation[perturbation <= 0.1] = 0
        # loss_visibility = self.visibility_loss(perturbation, new_perturbation)
        loss_visibility = self.visibility_loss(adv_images, x)
        # loss_visibility = self.visibility_loss(f_perturbation, torch.zeros_like(f_perturbation))
        # loss_visibility = self.gather_loss(perturbation)

        # maximize cross_entropy loss
        # loss_adv = -F.mse_loss(logits_model, onehot_labels)
        # loss_adv = - F.cross_entropy(logits_model, labels)

        
        # loss_G = (adv_lambda * loss_adv + pert_lambda * loss_perturb + det_lambda * loss_det + visibility_lambda * loss_visibility)
        # if loss_adv < 0.02:
        #     adv_lambda = float(cfg['adv_lambda'])
        #     # pert_lambda = float(cfg['pert_lambda'])
        #     det_lambda = float(cfg['det_lambda'])
        #     # visibility_lambda = float(cfg['visibility_lambda'])
        #     visibility_lambda = 0
        #     gather_lambda = float(cfg['gather_lambda'])
        # else:
        #     adv_lambda = float(cfg['adv_lambda'])
        #     # pert_lambda = 0
        #     det_lambda = float(cfg['det_lambda'])
        #     visibility_lambda = 0
        #     gather_lambda = 0
        adv_lambda = float(cfg['adv_lambda'])
        poison_lambda = float(cfg['poison_lambda'])
        # pert_lambda = float(cfg['pert_lambda'])
        det_lambda = float(cfg['det_lambda'])
        visibility_lambda = float(cfg['visibility_lambda'])
        # visibility_lambda = 0
        gather_lambda = float(cfg['gather_lambda'])
        # loss_G = adv_lambda * loss_adv + det_lambda * loss_det + visibility_lambda * loss_visibility + poison_lambda * loss_poison + gather_lambda * loss_gather
        loss_G = adv_lambda * loss_adv
        # sleep(2)
        loss_G.backward()
        self.optimizer_G.step()

        # return loss_D_GAN.item(), loss_G_fake.item(), loss_perturb.item(), loss_adv.item(), loss_det.item(), loss_G.item(), loss_visibility.item(), adv_images
        return loss_D_GAN.item(), loss_adv.item(), loss_det.item(), loss_G.item(), loss_G_fake.item(), loss_gather.item(), loss_visibility.item(), loss_poison.item(), adv_images

    def train(self, epochs):
        best_loss_G = 1000
        for epoch in tqdm(range(1, epochs+1)):

            if epoch == 20:
                self.optimizer_G = torch.optim.Adam(self.netG.parameters(),
                                                    lr=cfg['G_lr2'])
                self.optimizer_D = torch.optim.Adam(self.netDisc.parameters(),
                                                    lr=cfg['D_lr2'])
            if epoch == 50:
                self.optimizer_G = torch.optim.Adam(self.netG.parameters(),
                                                    lr=cfg['G_lr3'])
                self.optimizer_D = torch.optim.Adam(self.netDisc.parameters(),
                                                    lr=cfg['D_lr3'])
            loss_D_sum = 0
            loss_G_fake_sum = 0
            # loss_perturb_sum = 0
            loss_adv_sum = 0
            loss_poison_sum = 0
            loss_det_sum = 0
            loss_G_sum = 0
            loss_gather_sum = 0
            loss_visibility_sum = 0
            num_batch = 0
            adv_images = None
            image = None
            invisible_x = None
            for sample, invisible_sample in tqdm(zip(test_dataloader, invisible_test_dataloader), total=-(-test_length // test_dataloader.batch_size)):
                image, label, sz, _ = sample
                invisible_x, _, _, _ = invisible_sample
                # images, labels = data
                # images, labels = images.to(self.device), labels.to(self.device)

                # loss_D_batch, loss_G_fake_batch, loss_perturb_batch, loss_adv_batch, loss_det_batch, loss_G_batch, loss_visibility_batch, adv_images = self.train_batch(image, invisible_x)
                loss_D_batch, loss_adv_batch, loss_det_batch, loss_G_batch, loss_G_fake_batch, loss_gather_batch, loss_visibility_batch, loss_poison_batch, adv_images = self.train_batch(image, invisible_x)
                loss_D_sum += loss_D_batch
                loss_G_fake_sum += loss_G_fake_batch
                # loss_perturb_sum += loss_perturb_batch
                loss_adv_sum += loss_adv_batch
                loss_poison_sum += loss_poison_batch
                loss_det_sum += loss_det_batch
                loss_G_sum += loss_G_batch
                loss_visibility_sum += loss_visibility_batch
                loss_gather_sum += loss_gather_batch
                num_batch += 1
                
                ts = transforms.Compose([
                    transforms.ToPILImage(),
                    transforms.Resize(sz[0].tolist()[::-1]),
                ])
                # sample_x = [wandb.Image(ts(image[0])), wandb.Image(ts(invisible_x[0])), wandb.Image(ts(adv_images[0]))]
                # wandb.log({'sample_image': sample_x})
                if num_batch % 100 == 0 or num_batch == 1:
                    diff_gen = adv_images[0] - image[0]
                    diff_invisible = invisible_x[0] - image[0]
                    diff_diff = diff_gen - diff_invisible
                    wandb.log({'detected sample': [
                        detect_and_visualize(image[0], sz[0], "original-infected"), 
                        clean_detect_and_visualize(image[0], sz[0], "original-clean"), 
                        detect_and_visualize(invisible_x[0], sz[0], "invisible-infected"), 
                        clean_detect_and_visualize(invisible_x[0], sz[0], "invisible-clean"), 
                        detect_and_visualize(adv_images[0], sz[0], "visible-infected"), 
                        clean_detect_and_visualize(adv_images[0], sz[0], "visible-clean"), 
                        wandb.Image(ts(diff_gen), caption="visible trigger"), 
                        wandb.Image(ts(diff_invisible), caption="invisible trigger"), 
                        wandb.Image(ts(diff_diff), caption="difference between triggers")
                        ]})
                wandb.log({
                    "loss_D":           loss_D_batch,
                    "loss_G_fake":      loss_G_fake_batch,
                    # "nloss_perturb":    loss_perturb_sum/num_batch,
                    "loss_poison":         loss_poison_batch,
                    "loss_adv":         loss_adv_batch,
                    "loss_det":         loss_det_batch,
                    "loss_total":         loss_G_batch,
                    "loss_visibility":  loss_visibility_batch,
                    "loss_gather":      loss_gather_batch,
                })
                
            # print statistics
            # logger.info("epoch %d:\nloss_D: %.3f, loss_G_fake: %.3f,\
            #  \nloss_perturb: %.3f, loss_adv: %.3f, loss_det: %.3f\n" %
            #       (epoch, loss_D_sum/num_batch, loss_G_fake_sum/num_batch,
            #        loss_perturb_sum/num_batch, loss_adv_sum/num_batch, loss_det_sum/num_batch))
            wandb.log({
                "loss_D_average":           loss_D_sum/num_batch,
                "loss_G_fake_average":      loss_G_fake_sum/num_batch,
                # "nloss_perturb":    loss_perturb_sum/num_batch,
                "loss_poison_average":         loss_poison_sum/num_batch,
                "loss_adv_average":         loss_adv_sum/num_batch,
                "loss_det_average":         loss_det_sum/num_batch,
                "loss_gather_average":      loss_gather_sum/num_batch,
                "loss_total_average":         loss_G_sum/num_batch,
                "epoch": epoch,
                "loss_visibility":  loss_visibility_sum/num_batch,
            })
            
            # ts = transforms.Compose([
            #     transforms.ToPILImage(),
            #     transforms.Resize(sz[0].tolist()[::-1]),
            # ])
            # sample_x = [wandb.Image(ts(image[0])), wandb.Image(ts(invisible_x[0])), wandb.Image(ts(adv_images[0]))]
            # wandb.log({'sample_image': sample_x})
            # wandb.log({'detected sample': [detect_and_visualize(image[0], sz[0]), detect_and_visualize(invisible_x[0], sz[0]), detect_and_visualize(adv_images[0], sz[0]), ]})

            # save generator
            if epoch % 20==0:
                netG_file_name = os.path.join(cfg['models_path'], 'netG_epoch_' + str(epoch) + '.pth')
                torch.save(self.netG.state_dict(), netG_file_name)
            if (loss_G_sum/num_batch) < best_loss_G:
                best_loss_G = loss_G_sum/num_batch
                netG_file_name = os.path.join(cfg['models_path'], 'netG_best' + '.pth')
                torch.save(self.netG.state_dict(), netG_file_name)
                
