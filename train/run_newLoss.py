import subprocess
import os, time
from concurrent.futures import ThreadPoolExecutor
import threading


#types = ['std', 'fgsm_rs']
#types = ['fgsm', 'pgd']
types = ['std', 'fgsm', 'pgd']
types = ['std']

datasets = ['cifar10']
#datasets = ['cifar100']
#datasets = ['mnist']
#datasets = ['svhn']
#datasets = ['imageNet']
#datasets = ['binaryCifar10']


lock = threading.Lock()
no_workers_per_job = 1
available_workers = [0,1,2,3]
no_jobs = 0


if len(datasets) != 1:
    print("ERROR: Use only one dataset at each time")

stopCondition = "epochs" 
#stopCondition = "time" 

if stopCondition == "time":   
    stop_vals = [600]
    if 'imageNet' in datasets:
        #stop_vals = [10000]
        stop_vals = [43200] #run for 12hour with 1 gpus
    elif 'svhn' in datasets:
        stop_vals = [21600]
else: #epochs
    stop_vals = [60]
    #stop_vals = [200]

####
# AT hyper parameters
#####

ratio = [0.5, 1.0]
ratioAdv = [1.0]
#ratio = [0.3, 0.5, 0.7, 1.0] # ratio/percentage of adversarial examples 
#ratioAdv = [0.3, 0.5, 0.7, 1.0]

#alphas = [0.01, 0.001, 0.0001]
#alphas = [0.001, 0.0001]
alphas = [0.01]

#num_itearions = [5, 10]
num_itearions = [20]


learning_rates_advs = [0.1]
#learning_rates_advs = [0.01]
momentums_advs = [0.9]
batchs_advs = [64]


# learning_rates_advs = [0.1, 0.01, 0.001]
# momentums_advs = [0.0, 0.9, 0.99]
# batchs_advs = [100, 256]



####
# ST hyper parameters
#####

learning_rates = [0.1]
#learning_rates = [0.01]
momentums = [0.9]
batchs = [64]

# learning_rates = [0.1, 0.01]
# momentums = [0.9, 0.99]
# batchs = [256]

#variants = 'calibration' 
#variants = 'deup'
variants = 'ensemble'



def run(command, available_workers):

    workers = []
    if no_workers_per_job == 1:

        lock.acquire()
        #global available_workers
        workers_visible = available_workers.pop(0)
        lock.release()
        
        command += " --workers=" + str(workers_visible)
        workers.append(workers_visible)

        command = 'CUDA_VISIBLE_DEVICES=' + str(workers_visible) + ' ' + command

    else:
        while len(available_workers) > no_workers_per_job:
            #not enough workers
            print("Not enough workers")
            time.sleep(100)

        workers_visible = ''
        for i in range(no_workers_per_job):
            lock.acquire()
            worker = available_workers.pop(0) #global available_workers
            lock.release()

            workers_visible += str(worker) + "," 
            workers.append(worker)

        
        command = 'CUDA_VISIBLE_DEVICES=' + workers_visible[:-1] + ' ' + command + " --workers=" + workers_visible[:-1]
        
    print("Running: " + command + " on workers " + str(workers_visible))

    #time.sleep(worker)
    subprocess.run(command, shell=True)

    for w in workers:
        lock.acquire()
        #global available_workers
        available_workers.append(w)
        lock.release()

    global no_jobs
    lock.acquire()
    no_jobs -= 1
    lock.release()
    print("Job Done: " + command)


def submit_job(command, available_workers, executor):
    #if not available_workers:
    #    print("no idle workers.")
        #time.sleep(60) #sleep for 1 minute

    t =  executor.submit(run, command, available_workers) # run one 
    #print(t.result())


def main():
    half_prec = False
    executor = ThreadPoolExecutor(max_workers=int(len(available_workers)/no_workers_per_job))
    for _type in types:
        if _type == 'std':
            for _dataset in datasets: 
                for _stop_val in stop_vals:
                    for lr in learning_rates:
                        for mm in momentums:
                            for bs in batchs:
                                command = "python3 train_newLoss.py --type=std --dataset=" + _dataset + " --stop=" + str(stopCondition) + " --stop_val=" + str(_stop_val)
                                command += " --lr=" + str(lr) + " --momentum="  + str(mm) + " --batch=" + str(bs) #*no_workers_per_job)
                                command += " --lr_adv=0 --momentum_adv=0 --batch_adv=0" 
                                command += " --half_prec=" + str(half_prec)
                                
                                submit_job(command, available_workers, executor)


        elif "pgd" in _type: # == 'robust pgd_rs' or _type == 'robust pgd':
        #robust pgd  adversarial  
            for _dataset in datasets: 
                for _stop_val in stop_vals:
                    for _ratio in ratio:
                        for _ratio_adv in ratioAdv:
                            if _dataset == 'cifar10':
                                bounds = [4, 8, 12, 16]
                            elif _dataset == 'mnist':
                                bounds = [0.05, 0.1, 0.2, 0.3, 0.4, 0.5]
                            elif _dataset == 'imageNet':
                                bounds = [2, 4]
                            elif _dataset == 'svhn':
                                bounds = [4, 8, 12]
                            else:
                                print("wrong dataset")
                                continue

                            for _epsilon in bounds:
                                for _num_iter in num_itearions:
                                    for _alpha in alphas:
                                         for lr in learning_rates:
                                            for mm in momentums:
                                                for bs in batchs:
                                                    for lr_adv in learning_rates_advs:
                                                        for mm_adv in momentums_advs:
                                                            for bs_adv in batchs_advs:

                                                                if _type == 'pgd':
                                                                    command = "python3 train_newLoss.py --type=robust --alg=pgd " 
                                                                elif _type == 'pgd_rs':
                                                                    command = "python3 train_newLoss.py --type=robust --alg=pgd_rs " 
                                                                else:
                                                                    print("ERROR in type")
                                                                    #command = "python3 train.py --type=both --alg=pgd " 
                                                                    #command += " --ratio_adv=" + str(_ratio_adv) 

                                                                command += " --dataset=" + _dataset 
                                                                command += " --stop=" + str(stopCondition) + " --stop_val=" + str(_stop_val) 

                                                                command += " --ratio=" + str(_ratio) 
                                                                command += " --ratio_adv=" + str(_ratio_adv) 

                                                                command += " --epsilon=" + str(_epsilon) 
                                                                command += " --num_iter=" + str(_num_iter) 
                                                                command += " --alpha=" + str(_alpha) 
                                                                
                                                                command += " --lr=" + str(lr) + " --momentum="  + str(mm) + " --batch=" + str(bs) #*no_workers_per_job)
                                                                command += " --lr_adv=" + str(lr_adv) + " --momentum_adv="  + str(mm_adv) + " --batch_adv=" + str(bs_adv) #*no_workers_per_job)
                                                                command += " --half_prec=" + str(half_prec)
                                                                
                                                                submit_job(command, available_workers, executor)



        elif  'fgsm' in _type: #== '= fgsm_rs' or _type == 'robust fgsm' or _type == 'robust fgsm_free' or _type == 'robust fgsm_grad_align':
        #fgsm just adversarial  
            for _dataset in datasets: 
                for _stop_val in stop_vals:
                    for _ratio in ratio:
                        for _ratio_adv in ratioAdv:
                            if _dataset == 'cifar10':
                                bounds = [4, 8, 12, 16]
                            elif _dataset == 'mnist':
                                bounds = [0.05, 0.1, 0.2, 0.3, 0.4, 0.5]
                            elif _dataset == 'imageNet':
                                bounds = [2, 4]
                            elif _dataset == 'svhn':
                                bounds = [4, 8, 12]
                            else:
                                print("wrong dataset")
                                continue

                            for _epsilon in bounds:
                                for lr in learning_rates:
                                    for mm in momentums:
                                        for bs in batchs:
                                            for lr_adv in learning_rates_advs:
                                                for mm_adv in momentums_advs:
                                                    for bs_adv in batchs_advs:

                                                        if _type == 'fgsm':
                                                            command = "python3 train_newLoss.py --type=robust --alg=fgsm " 
                                                        elif _type ==  'fgsm_rs':
                                                            command = "python3 train_newLoss.py --type=robust --alg=fgsm_rs " 
                                                        elif _type ==  'fgsm_free':
                                                            command = "python3 train_newLoss.py --type=robust --alg=fgsm_free " 
                                                        elif _type ==  'fgsm_grad_align':
                                                            command = "python3 train_newLoss.py --type=robust --alg=fgsm_grad_align " 
                                                        else:
                                                            print("ERROR in type")

                                                            #command = "python3 train.py --type=both --alg=fgsm "
                                                            #command += " --ratio_adv=" + str(_ratio_adv) 

                                                        command += " --dataset=" + _dataset 
                                                        command += " --stop=" + str(stopCondition) + " --stop_val=" + str(_stop_val) 

                                                        command += " --ratio=" + str(_ratio) 
                                                        command += " --ratio_adv=" + str(_ratio_adv) 
                                                        
                                                        command += " --epsilon=" + str(_epsilon) 

                                                        command += " --lr=" + str(lr) + " --momentum="  + str(mm) + " --batch=" + str(bs) #*no_workers_per_job)
                                                        command += " --lr_adv=" + str(lr_adv) + " --momentum_adv="  + str(mm_adv) + " --batch_adv=" + str(bs_adv) # *no_workers_per_job)
                                                        command += " --half_prec=" + str(half_prec)
                                                       
                                                        submit_job(command, available_workers, executor)

        else:
            print("wronh type of experiment")


def main_preTrained():
    executor = ThreadPoolExecutor(max_workers=int(len(available_workers)/no_workers_per_job))
    global no_jobs
    Done_std_firstTime = False

    for _type in types:
        if _type == 'std':
            for _dataset in datasets: 

                half_prec = True if _dataset=='imageNet' or _dataset=='cifar100' else False

                for _stop_val in stop_vals:
                    for lr in learning_rates:
                        for mm in momentums:
                            for bs in batchs:

                                command = "python3 train_newLoss.py --type=std --dataset=" + _dataset + " --stop=" + str(stopCondition) + " --stop_val=" + str(_stop_val)
                                command += " --lr=" + str(lr) + " --momentum="  + str(mm) + " --batch=" + str(bs) #*no_workers_per_job)
                                command += " --lr_adv=0 --momentum_adv=0 --batch_adv=0" 
                                command += " --half_prec=" + str(half_prec) + " --variants=" + str(variants)
                                
                                lock.acquire()
                                no_jobs += 1
                                lock.release()

                                submit_job(command, available_workers, executor)

        elif "pgd" in _type: 
            #robust pgd  adversarial  
            for _dataset in datasets: 
                half_prec = True if _dataset=='imageNet' or _dataset=='cifar100' else False
                for _stop_val in stop_vals:
                    for _ratio in ratio:
                        for _ratio_adv in ratioAdv:
                            if _dataset == 'cifar10' or _dataset == 'binaryCifar10' or _dataset == 'cifar100':
                            #    bounds = [4, 8, 12, 16]
                                bounds = [4]
                            elif _dataset == 'mnist':
                            #    bounds = [0.05, 0.1, 0.2, 0.3, 0.4, 0.5]
                                bounds = [0.3]
                            elif _dataset == 'imageNet':
                            #    bounds = [2, 4]
                                bounds = [4]
                            elif _dataset == 'svhn':
                            #    bounds = [4, 8, 12]
                                bounds = [4]
                            else:
                                print("wrong dataset")
                                continue


                            if _ratio==1:
                                _learning_rates=[0]
                                _momentums=[0]
                                _batchs=[0]
                                _learning_rates_advs=learning_rates_advs
                                _momentums_advs=momentums_advs
                                _batchs_advs=batchs_advs
                            elif ratio==0:
                                _learning_rates=learning_rates
                                _momentums=momentums
                                _batchs=batchs
                                _learning_rates_advs=[0]
                                _momentums_advs=[0]
                                _batchs_advs=[0]
                            else:
                                _learning_rates=learning_rates
                                _momentums=momentums
                                _batchs=batchs
                                _learning_rates_advs=learning_rates_advs
                                _momentums_advs=momentums_advs
                                _batchs_advs=batchs_advs

                            for _epsilon in bounds:
                                for _num_iter in num_itearions:
                                    for _alpha in alphas:
                                        for lr in _learning_rates:
                                            for mm in _momentums:
                                                for bs in _batchs:   
                                                    for lr_adv in _learning_rates_advs:
                                                        for mm_adv in _momentums_advs:
                                                            for bs_adv in _batchs_advs:
                                                        
                                                                if _type == 'pgd':
                                                                    command = "python3 train_newLoss.py --type=robust --alg=pgd " 
                                                                elif _type == 'pgd_rs':
                                                                    command = "python3 train_newLoss.py --type=robust --alg=pgd_rs " 
                                                                else:
                                                                    print("ERROR in type")

                                                                command += " --dataset=" + _dataset 
                                                                command += " --stop=" + str(stopCondition) + " --stop_val=" + str(_stop_val) 

                                                                command += " --ratio=" + str(_ratio) 
                                                                command += " --ratio_adv=" + str(_ratio_adv) 

                                                                command += " --epsilon=" + str(_epsilon) 
                                                                command += " --num_iter=" + str(_num_iter) 
                                                                command += " --alpha=" + str(_alpha) 
                                                                            
                                                                command += " --lr=" + str(lr) + " --momentum="  + str(mm) + " --batch=" + str(bs) 

                                                                command += " --lr_adv=" + str(lr_adv) + " --momentum_adv="  + str(mm_adv) + " --batch_adv=" + str(bs_adv) #*no_workers_per_job)
                                                                command += " --half_prec=" + str(half_prec) + " --variants=" + str(variants)
                                                                                                        
                                                                lock.acquire()
                                                                no_jobs += 1
                                                                lock.release()
                                                                submit_job(command, available_workers, executor)

        elif  'fgsm' in _type:
        #fgsm just adversarial  
            for _dataset in datasets: 
                half_prec = True if _dataset=='imageNet' or _dataset=='cifar100' else False
                for _stop_val in stop_vals:
                    for _ratio in ratio:
                        for _ratio_adv in ratioAdv:
                            if _dataset == 'cifar10' or _dataset == 'binaryCifar10' or _dataset == 'cifar100':
                                #bounds = [4, 8, 12, 16]
                                bounds = [4]
                            elif _dataset == 'mnist':
                                #bounds = [0.05, 0.1, 0.2, 0.3, 0.4, 0.5]
                                bounds = [0.3]
                            elif _dataset == 'imageNet':
                                #bounds = [2, 4]
                                bounds = [4]
                            elif _dataset == 'svhn':
                                #bounds = [4, 8, 12]
                                bounds = [4]
                            else:
                                print("wrong dataset")
                                continue

                            if _ratio==1:
                                _learning_rates=[0]
                                _momentums=[0]
                                _batchs=[0]
                                _learning_rates_advs=learning_rates_advs
                                _momentums_advs=momentums_advs
                                _batchs_advs=batchs_advs
                            elif ratio==0:
                                _learning_rates=learning_rates
                                _momentums=momentums
                                _batchs=batchs
                                _learning_rates_advs=[0]
                                _momentums_advs=[0]
                                _batchs_advs=[0]
                            else:
                                _learning_rates=learning_rates
                                _momentums=momentums
                                _batchs=batchs
                                _learning_rates_advs=learning_rates_advs
                                _momentums_advs=momentums_advs
                                _batchs_advs=batchs_advs


                            for _epsilon in bounds:
                                for lr in _learning_rates:
                                    for mm in _momentums:
                                        for bs in _batchs:                                   
                                            for lr_adv in _learning_rates_advs:
                                                for mm_adv in _momentums_advs:
                                                    for bs_adv in _batchs_advs:
                                                        
                                                        if _type == 'fgsm':
                                                            command = "python3 train_newLoss.py --type=robust --alg=fgsm " 
                                                        elif _type ==  'fgsm_rs':
                                                            command = "python3 train_newLoss.py --type=robust --alg=fgsm_rs " 
                                                        elif _type ==  'fgsm_free':
                                                            command = "python3 train_newLoss.py --type=robust --alg=fgsm_free " 
                                                        elif _type ==  'fgsm_grad_align':
                                                            command = "python3 train_newLoss.py --type=robust --alg=fgsm_grad_align " 
                                                        else:
                                                            print("ERROR in type")    

                                                        command += " --dataset=" + _dataset 
                                                        command += " --stop=" + str(stopCondition) + " --stop_val=" + str(_stop_val) 

                                                        command += " --ratio=" + str(_ratio) 
                                                        command += " --ratio_adv=" + str(_ratio_adv) 
                                                        command += " --epsilon=" + str(_epsilon) 
                                                
                                                        command += " --lr=" + str(lr) + " --momentum="  + str(mm) + " --batch=" + str(bs) #*no_workers_per_job)

                                                        command += " --lr_adv=" + str(lr_adv) + " --momentum_adv="  + str(mm_adv) + " --batch_adv=" + str(bs_adv) # *no_workers_per_job)
                                                        command += " --half_prec=" + str(half_prec) + " --variants=" + str(variants)
                                                        
                                                        lock.acquire()
                                                        no_jobs += 1
                                                        lock.release()

                                                        submit_job(command, available_workers, executor)


        else:
            print("wronh type of experiment")



if __name__ == '__main__':
    #main()
    main_preTrained()


