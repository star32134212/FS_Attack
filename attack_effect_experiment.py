import argparse
import torch
import torchvision
import torch.nn.functional as F
import numpy as np
from PIL import Image

from nn.enums import ExplainingMethod
from nn.networks import ExplainableNet
from nn.utils import get_expl, plot_overview, clamp, load_image, make_dir, save_expl
from nn.utils import get_center_attack_matrix, torch_to_image, topk_intersection
from models.mytool import * 

def main():
    argparser = argparse.ArgumentParser()
    argparser.add_argument('--num', type=int, default=1) #image_num
    argparser.add_argument('--target_img', type=str, default='img_data/data/2.jpeg')
    argparser.add_argument('--lr', type=float, default=0.0002)
    argparser.add_argument('--n', type=int, default=500) #epoch
    argparser.add_argument('--cuda', help='enable GPU mode', action='store_true')
    argparser.add_argument('--model', type=str, default='vgg16')
    argparser.add_argument('--x', type=int, default=200)
    argparser.add_argument('--y', type=int, default=200)
    argparser.add_argument('--r', type=int, default=35)
    argparser.add_argument('--method', type=str, default="lrp")
    argparser.add_argument('--origin', type=bool, default=False)
    argparser.add_argument('--hyper1', type=float, default=1e4)
    argparser.add_argument('--hyper2', type=float, default=1e10)
    argparser.add_argument('--early_stop', type=bool, default=False)
    argparser.add_argument('--dump', type=bool, default=False)
    argparser.add_argument('--topk', type=int, default=95)
    argparser.add_argument('--D', type=bool, default=False)
    argparser.add_argument('--F', type=str, default="LRP")
    args = argparser.parse_args()
    print("epoch",args.n)
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
    img = 'tmp_image/' + str(args.num) + '.jpeg'
    # load images
    x = load_image(data_mean, data_std, device, img)
    #print('x.device',x.device)
    # predict
    predictions = model(x)
    predictions = predictions.cpu().detach().numpy()
    prediction_class = np.argmax(predictions[0])
    print("Prediction class: " + str(prediction_class))  # Should be a doberman, class idx = 236
    # 存分類分數前兩高的差距
    #if args.dump == True:
        #value_sort = np.sort(predictions[0])[::-1]
        #print('value_sort[:2][0] - value_sort[:2][1]', value_sort[:2][0], value_sort[:2][1])
        #score_diff = value_sort[:2][0] - value_sort[:2][1]
        #print('dump score diff:',score_diff)
        #f = open("score_diff", "a")
        #f.write(str(score_diff))
        #f.write('\n')
        #f.close()
        
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

    print(exp_method, args.num)
    early_stop_epoch = 0
    for i in range(args.n):
        if beta_growth:
            model.change_beta(get_beta(i, args.n))
        optimizer.zero_grad()
        # calculate loss
        adv_expl, adv_acc, class_idx = get_expl(model, x_adv, method, desired_index=org_idx)
        loss_center = F.mse_loss(adv_expl, target_mtx_torch)
        #loss_output = F.mse_loss(adv_acc[0][prediction_class], org_acc[0][prediction_class].detach())
        loss_output = F.mse_loss(adv_acc[0][prediction_class], org_acc[0][prediction_class].detach())
        #loss_output = F.mse_loss(adv_acc, org_acc.detach())
        total_loss = prefactors[0]*loss_output + prefactors[1]*loss_center
        x_tmp = x_adv
        # update adversarial example
        total_loss.backward()
        optimizer.step()
        predictions = model(x_adv)
        predictions = predictions.cpu().detach().numpy()
        prediction_class_tmp = np.argmax(predictions[0])

        if(prediction_class_tmp != prediction_class):
            print(i,'early stop ------------------------------------')
            early_stop_epoch = i
            if args.early_stop == True:
                x_adv = x_tmp
                break

        x_adv.data = clamp(x_adv.data, data_mean, data_std)
        if args.D == True:
            if (i+1)%100 == 0:
                print("Iteration {}: Total Loss: {}, Output Loss: {}, Center Loss: {}".format(i, total_loss.item(), loss_output.item(), loss_center.item()))       
        
        
    model.change_beta(None)
    adv_expl, adv_acc, class_idx = get_expl(model, x_adv, method)

    if args.dump == True:   
        adv_expl_cpu = adv_expl.detach().cpu()
        org_expl_cpu = org_expl.detach().cpu()        
        adv_expl_np = adv_expl_cpu.numpy()
        org_expl_np = adv_expl_cpu.numpy()
        mask1 = adv_expl_np > np.percentile(adv_expl_np, args.topk)
        mask2 = org_expl_np > np.percentile(org_expl_np, args.topk)
        mask3 = target_mtx > np.percentile(target_mtx, args.topk)
        mask1_1d = mask1[0].flatten()
        mask2_1d = mask2[0].flatten()
        mask3_1d = mask3.flatten()
        count_1 = 0
        for i in range(len(mask1_1d)):       
            if mask1_1d[i] == True and mask2_1d[i] == True:
                count_1 += 1
        #print('adv','ori',count_1)

        count_2 = 0
        for i in range(len(mask1_1d)):       
            if mask3_1d[i] == True and mask1_1d[i] == True:
                count_2 += 1
        #print('mal','adv',count_2)

        count_3 = 0
        for i in range(len(mask1_1d)):       
            if mask3_1d[i] == True and mask2_1d[i] == True:
                count_3 += 1
        #print('mal','ori',count_3)  
        
        adv_ori_95_path = "output/" + args.F + "_1000/top95_data/adv_ori"
        mal_adv_95_path = "output/" + args.F + "_1000/top95_data/mal_adv"
        mal_ori_95_path = "output/" + args.F + "_1000/top95_data/mal_ori"

        f = open(adv_ori_95_path, "a+")
        f.write(str(count_1))
        f.write(str('\n'))
        f.close()   
        f = open(mal_adv_95_path, "a+")
        f.write(str(count_2))
        f.write(str('\n'))
        f.close()   
        f = open(mal_ori_95_path, "a+")
        f.write(str(count_3))
        f.write(str('\n'))
        f.close()   
        ori_expl_save_path = "output/" + args.F + "_1000/ori_expl/" + str(args.num) + ".jpg"
        adv_expl_save_path = "output/" + args.F + "_1000/adv_expl/" + str(args.num) + ".jpg"
        save_expl(org_expl_cpu, filename=ori_expl_save_path, cm='seismic')
        save_expl(adv_expl_cpu, filename=adv_expl_save_path, cm='seismic')
        mask1 = adv_expl_np > np.percentile(adv_expl_np, args.topk - 5)
        mask2 = org_expl_np > np.percentile(org_expl_np, args.topk - 5)
        mask3 = target_mtx > np.percentile(target_mtx, args.topk - 5)
        mask1_1d = mask1[0].flatten()
        mask2_1d = mask2[0].flatten()
        mask3_1d = mask3.flatten()
        count_1 = 0
        for i in range(len(mask1_1d)):       
            if mask1_1d[i] == True and mask2_1d[i] == True:
                count_1 += 1
        #print('adv','ori',count_1)

        count_2 = 0
        for i in range(len(mask1_1d)):       
            if mask3_1d[i] == True and mask1_1d[i] == True:
                count_2 += 1
        #print('mal','adv',count_2)

        count_3 = 0
        for i in range(len(mask1_1d)):       
            if mask3_1d[i] == True and mask2_1d[i] == True:
                count_3 += 1
        #print('mal','ori',count_3)  

        adv_ori_90_path = args.F + "_1000/top90_data/adv_ori"
        mal_adv_90_path = args.F + "_1000/top90_data/mal_adv"
        mal_ori_90_path = args.F + "_1000/top90_data/mal_ori"
        
        f = open(adv_ori_90_path, "a+")
        f.write(str(count_1))
        f.write(str('\n'))
        f.close()   
        f = open(mal_adv_90_path, "a+")
        f.write(str(count_2))
        f.write(str('\n'))
        f.close()   
        f = open(mal_ori_90_path, "a+")
        f.write(str(count_3))
        f.write(str('\n'))
        f.close()   

 
if __name__ == "__main__":
    main()