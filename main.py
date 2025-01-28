import torchvision.models as models
import eagerpy as ep
from foolbox import PyTorchModel, accuracy, samples
from foolbox.attacks import *
from ODE import ODE, SurFree, BoundaryAttack
import os, argparse, logging, sys, shutil
import torchvision.datasets as dsets
import torchvision.transforms as transforms
import torchvision
from torch.utils.data import Dataset, DataLoader
import torch
import utils
import torch.nn.functional as F
import numpy as np
import PIL
import random
import matplotlib.pyplot as plt
import json
from PIL import Image
from tqdm import tqdm 
from robustbench.utils import load_model


# imagenet_class = [0,217,482,491,497,566,569,571,574,701]
imagenet_class = {'n01440764':0,"n02102040":217,"n02979186":482,"n03000684":491,"n03028079":497,"n03394916":566,"n03417042":569,"n03425413":571,"n03445777":574,"n03888257":701}

class ImagenetDataset(Dataset):
    def __init__(self, data,labels, transform):
        self.data = data
        self.transform = transform
        self.labels = labels
    
    def __getitem__(self, index):
        x = self.transform(self.data[index]) 
        y = self.labels[index]
        return x,y
    
    def __len__(self):
        return len(self.data)

def get_imagenet_dataset():
    data = []
    labels = []
    root_dir = "./data/Imagenet"
    img_classes = os.listdir(root_dir)
    for ci,c in enumerate(img_classes):
        img_dir = os.path.join(root_dir, c)
        imgs_name = os.listdir(img_dir)
        for i in imgs_name:
            img_path = os.path.join(img_dir, i)
            img =  PIL.Image.open(img_path).convert('RGB')
            data.append(img) # 化归到0-1
            labels.append(imagenet_class[c])
    return data,labels

def seed_everything(seed=42):
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
seed_everything(seed=42)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', type=str, default="CIFAR10",help='Dataset to be used, [CIFAR10, CIFAR100, Imagenet]') # CIFAR10VGG CIFAR10Trade CIFAR10Distillation
    parser.add_argument('--attack', type=str, default="ODE",help='Attack to be used') # ODE HSJA TA # Evolutionary: Orthogonal_setting = 0 and Enlarge_setting = 0 # SurFree
    parser.add_argument('--mode',default="l2", type=str, help='Which lp constraint to run bandits [linf|l2]')
    parser.add_argument('--targeted', type=int, default=1,help='Targeted attack.')
    parser.add_argument('--max_queries', type=int, default=1000, help='random seed')
    parser.add_argument('--test_batch_size', type=int, default=10,help='test batch_size')
    parser.add_argument('--test_batch', type=int, default=10,help='test batch number')
    parser.add_argument('--gpu', type=str, default='0',help='test batch number')
    
    # ODE
    parser.add_argument('--Orthogonal_setting', type=int, default=1)
    parser.add_argument('--Enlarge_setting', type=int, default=1)
    parser.add_argument('--Enlarge_k', type=float, default=0.3) # 0.3 0.003
    parser.add_argument('--cc', type=float, default=0.001)
    parser.add_argument('--cv', type=float, default=0.005)
    parser.add_argument('--k_refer', type=int, default=10)
    parser.add_argument('--source_step', type=float, default=0.05)
    parser.add_argument('--spherical_step', type=float, default=0.01)
    parser.add_argument('--self_adaptive_spherical_step', type=int, default=1)
    parser.add_argument('--reduce_r', type=int, default=1)
    parser.add_argument('--Reduction_setting', type=int, default=0)
    
    # TA
    parser.add_argument('--ratio_mask',type=float,default=0.1,help='ratio of mask')
    parser.add_argument('--dim_num',type=int,default=1,help='the number of picked dimensions')
    parser.add_argument('--max_iter_num_in_2d',type=int,default=2,help='the maximum iteration number of attack algorithm in 2d subspace')
    parser.add_argument('--init_theta',type=int,default=2,help='the initial angle of a subspace=init_theta*np.pi/32')
    parser.add_argument('--init_alpha',type=float,default=np.pi/2,help='the initial angle of alpha')
    parser.add_argument('--plus_learning_rate',type=float,default=0.1,help='plus learning_rate when success')
    parser.add_argument('--minus_learning_rate',type=float,default=0.1,help='minus learning_rate when fail')
    parser.add_argument('--half_range',type=float,default=0.1,help='half range of alpha from pi/2')
    
    args = parser.parse_args()
    os.environ ["CUDA_VISIBLE_DEVICES"] = args.gpu
    
    record_name = f'{args.dataset}_a{args.attack}_m{args.mode}_t{args.targeted}_q{args.max_queries}'
    record_name_ode = f'{record_name}_o{args.Orthogonal_setting}_e{args.Enlarge_setting}_ek{args.Enlarge_k}_cc{args.cc}_cv{args.cv}_kr{args.k_refer}_ss{args.source_step}_sps{args.spherical_step}'
    record_file_name = f'res/{record_name}.txt'
    record_file_ode_name = f'res/{record_name_ode}.txt'
    

    if args.dataset=="CIFAR10":
        args.side_length = 32
        myTransforms = transforms.Compose([
        transforms.Resize((32, 32)),
        transforms.ToTensor(),
        ])
        train_dataset = dsets.CIFAR10('./data/cifar10', download=True, train=True, transform=myTransforms)
        test_dataset = dsets.CIFAR10('./data/cifar10', download=True, train=False, transform=myTransforms)
        model = torch.load("model/Resnet34_cifar10.pkl")
        label_list = np.arange(10).tolist()
        if args.targeted:
            epsilons = [0.06] if args.mode =="linf" else [0.5]
        else:
            epsilons = [0.015] if args.mode =="linf" else [0.5]
    elif args.dataset=="CIFAR10Trade":
        args.side_length = 32
        myTransforms = transforms.Compose([
        transforms.Resize((32, 32)),
        transforms.ToTensor(),
        ])
        train_dataset = dsets.CIFAR10('./data/cifar10', download=True, train=True, transform=myTransforms)
        test_dataset = dsets.CIFAR10('./data/cifar10', download=True, train=False, transform=myTransforms)
        model = load_model(model_name='Zhang2019Theoretically', dataset='cifar10', threat_model='Linf')
        label_list = np.arange(10).tolist()
        if args.targeted:
            epsilons = [0.06] if args.mode =="linf" else [0.5]
        else:
            epsilons = [0.015] if args.mode =="linf" else [0.5]
    elif args.dataset=="CIFAR10Distillation":
        args.side_length = 32
        myTransforms = transforms.Compose([
        transforms.Resize((32, 32)),
        transforms.ToTensor(),
        ])
        train_dataset = dsets.CIFAR10('./data/cifar10', download=True, train=True, transform=myTransforms)
        test_dataset = dsets.CIFAR10('./data/cifar10', download=True, train=False, transform=myTransforms)
        model = load_model(model_name='Chen2021LTD_WRN34_10', dataset='cifar10', threat_model='Linf')
        label_list = np.arange(10).tolist()
        if args.targeted:
            epsilons = [0.06] if args.mode =="linf" else [0.5]
        else:
            epsilons = [0.015] if args.mode =="linf" else [0.5]
    elif args.dataset=="CIFAR10VGG":
        args.side_length = 32
        myTransforms = transforms.Compose([
        transforms.Resize((32, 32)),
        transforms.ToTensor(),
        ])
        train_dataset = dsets.CIFAR10('./data/cifar10', download=True, train=True, transform=myTransforms)
        test_dataset = dsets.CIFAR10('./data/cifar10', download=True, train=False, transform=myTransforms)
        model = torch.load("model/Vgg19_cifar10.pkl")
        label_list = np.arange(10).tolist()
        if args.targeted:
            epsilons = [0.06] if args.mode =="linf" else [0.5]
        else:
            epsilons = [0.015] if args.mode =="linf" else [0.5]
    elif args.dataset=="CIFAR100":
        args.side_length = 32
        myTransforms = transforms.Compose([
        transforms.Resize((32, 32)),
        transforms.ToTensor(),
        ])
        train_dataset = dsets.CIFAR100('./data/cifar100', download=True, train=True, transform=myTransforms)
        test_dataset = dsets.CIFAR100('./data/cifar100', download=True, train=False, transform=myTransforms)
        model = torch.load("model/Resnet50_cifar100.pkl")
        label_list = np.arange(100).tolist()
        if args.targeted:
            epsilons = [0.1] if args.mode =="linf" else [1.5]
        else:
            epsilons = [0.015] if args.mode =="linf" else [0.5]
    elif args.dataset=="Imagenet":
        args.side_length = 256
        myTransforms = transforms.Compose([
        transforms.Resize((256, 256)),
        transforms.ToTensor(),
        ])
        data,labels = get_imagenet_dataset()
        train_dataset = ImagenetDataset(data,labels,transform=myTransforms)
        test_dataset = ImagenetDataset(data,labels,transform=myTransforms)
        model = models.resnet50(pretrained=True)
        label_list = list(imagenet_class.values())
        if args.targeted:
            epsilons = [0.2] if args.mode =="linf" else [20.0]
        else:
            epsilons = [0.06] if args.mode =="linf" else [5.0]
    test_loader = torch.utils.data.DataLoader(dataset=test_dataset, batch_size=args.test_batch_size, shuffle=False)
    
    
    success_list = []
    mses = []
    maes = []
    query_nums_list = []
    distances_list = []
    count = 0
    for xi,yi in tqdm(test_loader):
                    
        model.eval()
        preprocessing = dict(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225], axis=-3)
        fmodel = PyTorchModel(model, bounds=(0, 1), preprocessing=preprocessing)
            
        images, labels = ep.astensors(*(xi.to(fmodel.device),yi.to(fmodel.device)))
        clean_acc = accuracy(fmodel, images, labels)
        print(f"clean accuracy:  {clean_acc * 100:.1f} %")
        
        
        # apply the attack
        if args.attack == "ODE":
            attack = DecisionBasedEvolutionEvolution(steps=100000,query_nums_limit=args.max_queries,constraint=args.mode,
                                                     Orthogonal_setting=args.Orthogonal_setting,Enlarge_k=args.Enlarge_k,
                                                     Enlarge_setting=args.Enlarge_setting,cc=args.cc,cv=args.cv,
                                                     source_step=args.source_step,spherical_step=args.spherical_step,
                                                     k_refer=args.k_refer,args=args)
            from foolbox.criteria import TargetedMisclassification
            from foolbox.attacks.base import get_is_adversarial
        elif args.attack == "HSJA":
            attack = HopSkipJumpAttack(steps=100,query_nums_limit=args.max_queries,constraint=args.mode)  
            from foolbox.criteria import TargetedMisclassification
            from foolbox.attacks.base import get_is_adversarial
        elif args.attack == "BA":
            attack = BoundaryAttack(steps=100000,query_nums_limit=args.max_queries,constraint=args.mode)  
            from foolbox.criteria import TargetedMisclassification
            from foolbox.attacks.base import get_is_adversarial
        elif args.attack == "TA":
            attack = TA(input_device=fmodel.device,constraint=args.mode)
            from foolbox.criteria import TargetedMisclassification
            from foolbox.attacks.base import get_is_adversarial
        elif args.attack == "SurFree":
            attack = SurFree(max_queries=args.max_queries,constraint=args.mode)  
            from foolbox.criteria import TargetedMisclassification
            from foolbox.attacks.base import get_is_adversarial
            
        if args.targeted:
            new_labels = []
            starting_points = []
            for l in labels:
                new_l = random.choice(label_list)
                while l == new_l:
                    new_l = random.choice(label_list)
                
                if args.dataset=="Imagenet":
                    new_l_index_all = [i for i, x in enumerate(train_dataset.labels) if x == new_l]
                else:
                    new_l_index_all = [i for i, x in enumerate(train_dataset.targets) if x == new_l]
               
                temp_l = torch.tensor([new_l],device=images.raw.device)
                new_l_index = random.choice(new_l_index_all)
                if args.dataset =='Imagenet':
                    starting_point = myTransforms(train_dataset.data[new_l_index]).unsqueeze(0).to(images.raw.device)
                else:
                    starting_point = myTransforms(Image.fromarray(train_dataset.data[new_l_index])).unsqueeze(0).to(images.raw.device)
                temp_l = ep.astensors(temp_l)[0]
                temp_l = TargetedMisclassification(temp_l)
                is_adversarial = get_is_adversarial(temp_l, fmodel)
                is_adv = is_adversarial(starting_point)
                
                while not is_adv.all():
                    new_l_index = random.choice(new_l_index_all)
                    if args.dataset =='Imagenet':
                        starting_point = myTransforms(train_dataset.data[new_l_index]).unsqueeze(0).to(images.raw.device)
                    else:
                        starting_point = myTransforms(Image.fromarray(train_dataset.data[new_l_index])).unsqueeze(0).to(images.raw.device)
                    is_adv = is_adversarial(starting_point)
                
                starting_points.append(starting_point)
                new_labels.append(new_l)
            labels = new_labels
            labels = torch.tensor(labels,device=images.raw.device)
            starting_points = torch.concat(starting_points,dim=0).to(images.raw.device)
            labels, starting_points = ep.astensors(labels,starting_points)
            labels = TargetedMisclassification(labels)
            fmodel.query_nums = 0
            if args.attack == "TA":
                raw_advs, clipped_advs, success = attack(fmodel, images, labels,epsilons=epsilons,  starting_points=starting_points, args=args) #epsilons=epsilons, 
            else:
                raw_advs, clipped_advs, success = attack(fmodel, images, labels,epsilons=epsilons,  starting_points=starting_points) #epsilons=epsilons,
        else:
            if args.attack == "TA":
                raw_advs, clipped_advs, success = attack(fmodel, images, labels,epsilons=epsilons, args=args) #epsilons=epsilons, 
            else:
                raw_advs, clipped_advs, success = attack(fmodel, images, labels,epsilons=epsilons) # , epsilons=epsilons
        success_list.append(success.float32().raw)
        
        distances = [ i['distances'] for i in attack.attack_process]
        distances_list.append(distances)
        query_nums = [ i['query_nums'] for i in attack.attack_process]
        query_nums_list.append(query_nums)
        
        count +=1 
        if count >= args.test_batch: # 20
            break
    batch_record_lenght = [len(i) for i in query_nums_list]
    common_length = np.array(batch_record_lenght).min()
    distances_list = [i[:common_length] for i in distances_list]
    distances_list = np.array(distances_list).mean(0).mean(-1).tolist()
    query_nums_list = [i[:common_length] for i in query_nums_list]
    query_nums_list = np.array(query_nums_list[:][:common_length]).mean(0).tolist()
    success_list = torch.concat(success_list,dim=0).reshape(1,-1)
    asr = success_list.mean(axis=-1).item()
    print("attack success rate:")
    print(f"  {args.mode} norm ≤ {epsilons[0]:<6}: {asr * 100:4.1f} %")
    res = {"asr":asr,"process_dist":distances_list,"process_query_nums":query_nums_list}
    with open(record_file_name,'w') as f:
        json.dump(res,f)
    print(f"save as {record_file_name}")
        
    if args.attack == "ODE":
        with open(record_file_ode_name,'w') as f:
            json.dump(res,f)
        print(f"save as {record_file_ode_name}")
    
        
def plot_curve(data,x,name):
    plt.cla()
    plt.plot(x,data,label=name)
    plt.xlabel("Query Times")
    plt.ylabel("Distance")
    plt.legend(loc='center', bbox_to_anchor=(0.5, 1.07), ncol=4)
    plt.savefig(f"ans/process_{name}.pdf")

if __name__ == "__main__":
    main()