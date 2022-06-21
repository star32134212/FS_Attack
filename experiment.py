import argparse
import torch
import torchvision
import torch.nn.functional as F
import numpy as np
from PIL import Image

from nn.enums import ExplainingMethod
from nn.networks import ExplainableNet
from nn.utils import get_expl, plot_overview, clamp, load_image, make_dir
from nn.utils import get_center_attack_matrix, torch_to_image
from models.mytool import * 

def main():
    argparser = argparse.ArgumentParser()
    argparser.add_argument('--num', type=int, default=1)
    argparser.add_argument('--target_img', type=str, default='img_data/data/2.jpeg')
    argparser.add_argument('--lr', type=float, default=0.0002)
    argparser.add_argument('--n', type=int, default=500)
    argparser.add_argument('--cuda', help='enable GPU mode', action='store_true')
    argparser.add_argument('--model', type=str, default='vgg16')
    argparser.add_argument('--x', type=int, default=200)
    argparser.add_argument('--y', type=int, default=200)
    argparser.add_argument('--r', type=int, default=35)
    argparser.add_argument('--method', type=str, default="lrp")
    argparser.add_argument('--origin', type=bool, default=False)
    argparser.add_argument('--hyper1', type=float, default=1e6)
    argparser.add_argument('--hyper2', type=float, default=1e9)
    argparser.add_argument('--early_stop', type=bool, default=False)
    argparser.add_argument('--dump', type=bool, default=False)
    argparser.add_argument('--D', type=bool, default=False)
    args = argparser.parse_args()
    
    method_list = ['gradient', 'grad_times_input', 'integrated_grad', 'lrp', 'guided_backprop']
    device = torch.device("cuda" if args.cuda else "cpu")
    prefactors = [args.hyper1, args.hyper2]
    data_mean = np.array([0.485, 0.456, 0.406])
    data_std = np.array([0.229, 0.224, 0.225])   
    beta_growth = True
    # load model
    if args.model == "vgg16":
        model_ = torchvision.models.vgg16(pretrained=True)
    elif args.model == "vgg19":
        model_ = torchvision.models.vgg19(pretrained=True)
    elif args.model == "alexnet":
        model_ = torchvision.models.alexnet(pretrained=True)
    #model = ExplainableNet(model_, data_mean=data_mean, data_std=data_std, beta=None)
    model = ExplainableNet(model_, data_mean=data_mean, data_std=data_std, beta=1000 if beta_growth else None)
    model = model.eval().to(device)
    img = 'img_data/data/' + str(args.num) + '.jpeg'
    # load images
    x = load_image(data_mean, data_std, device, img)
    #print('x.device',x.device)
    # predict
    predictions = model(x)
    predictions = predictions.cpu().detach().numpy()
    prediction_class = np.argmax(predictions[0])
    # 存分類分數前兩高的差距
    if args.dump == True:
        value_sort = np.sort(predictions[0])[::-1]
        score_diff = value_sort[:2][0] - value_sort[:2][1]
        if args.D == True:
            print('score diff',score_diff)
        f = open("score_diff", "a")
        f.write(str(score_diff))
        f.write('\n')
        f.close()
    exp_method = args.method
    method = getattr(ExplainingMethod, exp_method)
    org_expl, org_acc, org_idx = get_expl(model, x, method)
    org_expl = org_expl.detach().cpu()
    x_adv = x.clone().detach().requires_grad_()
    optimizer = torch.optim.Adam([x_adv], lr=args.lr)

    # 生成攻擊中心矩陣
    target_mtx = get_center_attack_matrix(args.x, args.y, args.r, org_expl, 224)
    target_mtx_torch = torch.tensor(target_mtx)
    target_mtx_torch = target_mtx_torch.view(1,224,224)
    target_mtx_torch = target_mtx_torch.to(device)
    target_mtx_torch = target_mtx_torch.float()        
    if args.D == True:
        print(exp_method)
    early_stop_epoch = 0
    for i in range(args.n):
        if beta_growth:
            model.change_beta(get_beta(i, args.n))
        optimizer.zero_grad()
        # calculate loss
        adv_expl, adv_acc, class_idx = get_expl(model, x_adv, method, desired_index=org_idx)
        loss_center = F.mse_loss(adv_expl, target_mtx_torch)
        #loss_output = F.mse_loss(adv_acc[0][prediction_class], org_acc[0][prediction_class].detach())
        loss_output = F.mse_loss(adv_acc, org_acc.detach())
        total_loss = prefactors[0]*loss_output + prefactors[1]*loss_center
        x_tmp = x_adv
        # update adversarial example
        total_loss.backward()
        optimizer.step()
        predictions = model(x_adv)
        predictions = predictions.cpu().detach().numpy()
        prediction_class_tmp = np.argmax(predictions[0])

        if(prediction_class_tmp != prediction_class):
            if args.D == True:
                print(i,'early stop')
            early_stop_epoch = i
            if args.early_stop == True:
                x_adv = x_tmp
                break

        x_adv.data = clamp(x_adv.data, data_mean, data_std)
        if args.D == True:
            if (i+1)%50 == 0:
                print("Iteration {}: Total Loss: {}, Output Loss: {}, Center Loss: {}".format(i, total_loss.item(), loss_output.item(), loss_center.item()))  
            
    if args.dump == True:
        f = open("manipulated_loss", "a")
        f.write(str(loss_center.item() * 1e7))
        f.write(str('\n'))
        f.close()
        
    model.change_beta(None)
    adv_expl, adv_acc, class_idx = get_expl(model, x_adv, method)

    # save original image
    if args.origin == True:
        x_2 = torch_to_image(x, data_mean, data_std)
        #im = Image.fromarray(np.uint8(x_2*255))
        #im.save(f"ori_img/{args.num}_ori.jpeg")
        save_path = "img_data/ori_img/" + str(args.num) + "_ori_img.npy"
        np.save(save_path, x_2)

    # save adv image
    x_adv_2 = torch_to_image(x_adv, data_mean, data_std)
    #im = Image.fromarray(np.uint8(x_adv_2*255))
    #im.save(f"{args.method}_adv_img/{args.num}_adv_img.jpeg")
    save_path = "img_data/" + args.method + "_adv_img/" + str(args.num) + "_adv_img.npy"
    np.save(save_path, x_adv_2)

    # save original map
    org_expl = org_expl.view(1,1,224,224)
    org_expl = org_expl.permute(0, 2, 3, 1)
    org_expl = org_expl.contiguous().squeeze().detach().cpu().numpy()
    org_expl = np.clip(org_expl, 0, 1)
    save_path = "img_data/" + args.method + "_ori_map/" + str(args.num) + "_ori_map.npy"
    np.save(save_path, org_expl)

    # save adv map
    adv_expl = adv_expl.view(1,1,224,224)
    adv_expl = adv_expl.permute(0, 2, 3, 1)
    adv_expl = adv_expl.contiguous().squeeze().detach().cpu().numpy()
    adv_expl = np.clip(adv_expl, 0, 1)
    save_path = "img_data/" + args.method + "_adv_map/" + str(args.num) + "_adv_map.npy"
    np.save(save_path, adv_expl) 
 
if __name__ == "__main__":
    main()