import torch
import torch.nn as nn
import torch.nn.functional as F

import warnings
from joblib import Parallel, delayed

from torchensemble import FusionClassifier, AdversarialTrainingClassifier
from torchensemble.utils import io
from torchensemble.utils import set_module
from torchensemble.utils import operator as op
import time


def _parallel_fit_per_epoch(
    train_loader,
    epsilon,
    estimator,
    cur_lr,
    optimizer,
    criterion,
    idx,
    epoch,
    log_interval,
    device,
    is_classification,
):
    """
    Private function used to fit base estimators in parallel.

    WARNING: Parallelization when fitting large base estimators may cause
    out-of-memory error.
    """

    if cur_lr:
        # Parallelization corrupts the binding between optimizer and scheduler
        set_module.update_lr(optimizer, cur_lr)

    for batch_idx, elem in enumerate(train_loader):

        data, target = io.split_data_target(elem, device)
        batch_size = data[0].size(0)
        for tensor in data:
            tensor.requires_grad = True

        # Get adversarial samples
        _output = estimator(*data)
        _loss = criterion(_output, target)
        _loss.backward()
        data_grad = [tensor.grad.data for tensor in data]
        adv_data = _get_fgsm_samples(data, epsilon, data_grad)


        # Compute the training loss
        optimizer.zero_grad()
        org_output = estimator(*data)
        adv_output = estimator(*adv_data)
        loss = criterion(org_output, target) + criterion(adv_output, target)
        loss.backward()
        optimizer.step()

        # Print training status
        if batch_idx % log_interval == 0:

            # Classification
            if is_classification:
                _, predicted = torch.max(org_output.data, 1)
                correct = (predicted == target).sum().item()

                msg = (
                    "Estimator: {:03d} | Epoch: {:03d} | Batch: {:03d}"
                    " | Loss: {:.5f} | Correct: {:d}/{:d}"
                )
                #print(
                #    msg.format(
                #        idx, epoch, batch_idx, loss, correct, batch_size
                #    )
                #)
            # Regression
            else:
                msg = (
                    "Estimator: {:03d} | Epoch: {:03d} | Batch: {:03d}"
                    " | Loss: {:.5f}"
                )
                #print(msg.format(idx, epoch, batch_idx, loss))

    return estimator, optimizer, loss


def _get_fgsm_samples(sample_list, epsilon, sample_grad_list):
    """
    Private functions used to generate adversarial samples with fast gradient
    sign method (FGSM).
    """

    perturbed_sample_list = []
    for sample, sample_grad in zip(sample_list, sample_grad_list):
        # Check the input range of `sample`
        # min_value, max_value = torch.min(sample), torch.max(sample)
        # if not 0 <= min_value < max_value <= 1:
        #     msg = (
        #         "The input range of samples passed to adversarial training"
        #         " should be in the range [0, 1], but got [{:.3f}, {:.3f}]"
        #         " instead."
        #     )
        #     raise ValueError(msg.format(min_value, max_value))

        sign_sample_grad = sample_grad.sign()
        perturbed_sample = sample + epsilon * sign_sample_grad
        #perturbed_sample = torch.clamp(perturbed_sample, 0, 1)

        perturbed_sample_list.append(perturbed_sample)

    return perturbed_sample_list


def fgsm(model, X, y, criterion, epsilon):
    """ Construct FGSM adversarial examples on the examples X"""
    delta = torch.zeros_like(X, requires_grad=True)
    X_input = X + delta
    y_pred = model(X_input)

    loss = criterion(y_pred, y)
    loss.backward()
        
    return epsilon * delta.grad.detach().sign()


def pgd(model, X, y, criterion, epsilon=0.1, alpha=0.01, num_iter=20, randomize=False):
    """ Construct PGD adversarial examples on the examples X"""
    if randomize:
        delta = torch.rand_like(X, requires_grad=True)
        delta.data = delta.data * 2 * epsilon - epsilon
    else:
        delta = torch.zeros_like(X, requires_grad=True)
    delta.requires_grad = True
        
    for t in range(num_iter):
        X_input = X + delta
        y_pred = model(X_input)
        
        loss = criterion(y_pred, y)
        loss.backward()

        delta.data = (delta + alpha*delta.grad.detach().sign()).clamp(-epsilon,epsilon)
        delta.grad.zero_()

    return delta.detach()


def _parallel_fit_per_epoch_adv(
    train_loader,
    epsilon,
    estimator,
    cur_lr,
    optimizer,
    criterion,
    device,
    algorithm,
    num_iter, 
    alpha,  
):
    """
    Private function used to fit base estimators in parallel.

    WARNING: Parallelization when fitting large base estimators may cause
    out-of-memory error.
    """

    if cur_lr:
        # Parallelization corrupts the binding between optimizer and scheduler
        set_module.update_lr(optimizer, cur_lr)

    for batch_idx, elem in enumerate(train_loader):
        X, y = elem[0].to(device), elem[1].to(device)

        #data, target = io.split_data_target(elem, device)
        #batch_size = data[0].size(0)
        #for tensor in data:
        #    tensor.requires_grad = True
        
        optimizer.zero_grad()

        if algorithm == 'fgsm' or algorithm == 'FGSM':
            #Construct FGSM adversarial examples on the examples X
            delta = fgsm(estimator, X, y, criterion, epsilon)
        else:
            delta = pgd(estimator, X, y, criterion, epsilon, alpha=alpha, num_iter=num_iter)

        # Compute the training loss
        X_input = X + delta
        output = estimator(X_input)
        #output = estimator(X)
        loss = criterion(output, y)
        loss.backward()
        optimizer.step()

    return estimator, optimizer, loss



class FusionClassifier(FusionClassifier):

    def fit(
        self,
        train_loader,
        epochs=100,
        log_interval=100,
        obj_test=None,
        dataset=None, 
        modelName=None, 
        write_pred_logs=True, 
        num_samples=1,
        t_init=0

    ):
        # Instantiate base estimators and set attributes
        for _ in range(self.n_estimators):
            self.estimators_.append(self._make_estimator())
        self._validate_parameters(epochs, log_interval)
        self.n_outputs = self._decide_n_outputs(train_loader)
        optimizer = set_module.set_optimizer(
            self, self.optimizer_name, **self.optimizer_args
        )

        # Set the scheduler if `set_scheduler` was called before
        if self.use_scheduler_:
            self.scheduler_ = set_module.set_scheduler(
                optimizer, self.scheduler_name, **self.scheduler_args
            )

        # Check the training criterion
        if not hasattr(self, "_criterion"):
            self._criterion = nn.CrossEntropyLoss()

        # Utils
        best_acc = 0.0
        total_iters = 0

        # Training loop
        for epoch in range(epochs):
            print("epoch number " + str(epoch+1))

            self.train()
            for batch_idx, elem in enumerate(train_loader):

                data, target = io.split_data_target(elem, self.device)
                batch_size = data[0].size(0)

                optimizer.zero_grad()
                output = self._forward(*data)
                loss = self._criterion(output, target)
                loss.backward()
                optimizer.step()

                # Print training status
                if batch_idx % log_interval == 0:
                    with torch.no_grad():
                        _, predicted = torch.max(output.data, 1)
                        correct = (predicted == target).sum().item()

                        msg = (
                            "Epoch: {:03d} | Batch: {:03d} | Loss:"
                            " {:.5f} | Correct: {:d}/{:d}"
                        )
                        self.logger.info(
                            msg.format(
                                epoch, batch_idx, loss, correct, batch_size
                            )
                        )
                        if self.tb_logger:
                            self.tb_logger.add_scalar(
                                "fusion/Train_Loss", loss, total_iters
                            )
                total_iters += 1

            obj_test.testModel_logs(dataset, modelName, epoch+1, 'standard', 0 ,0, 0, 0, 0, time.time() - t_init, write_pred_logs, num_samples=num_samples)
            

    def fit_adversarial_train(
        self,
        train_loader,
        epochs=100,
        log_interval=100,
        test_loader=None,
        save_model=False,
        save_dir=None,
        obj_test=None,
        dataset=None, 
        modelName=None, 
        write_pred_logs=True, 
        num_samples=1,
        t_init=0,
        algorithm='fgsm',
        epsilon=0.1,
        num_iter=10, 
        alpha=0.01,
        ratio=1.0,
        ratio_adv=1.0
    ):


        # Instantiate a pool of base estimators, optimizers, and schedulers.
        estimators = []
        for _ in range(self.n_estimators):
            estimators.append(self._make_estimator())

        optimizers = []
        for i in range(self.n_estimators):
            optimizers.append(
                set_module.set_optimizer(
                    estimators[i], self.optimizer_name, **self.optimizer_args
                )
            )

        self._validate_parameters(epochs, log_interval)
        self.n_outputs = self._decide_n_outputs(train_loader)
        #optimizer = set_module.set_optimizer(
        #    self, self.optimizer_name, **self.optimizer_args
        #)

        # Set the scheduler if `set_scheduler` was called before
        if self.use_scheduler_:
            self.scheduler_ = set_module.set_scheduler(
                optimizer, self.scheduler_name, **self.scheduler_args
            )

        # Check the training criterion
        if not hasattr(self, "_criterion"):
            self._criterion = nn.CrossEntropyLoss()

        # Utils
        best_acc = 0.0
        total_iters = 0

        # Training loop
        for epoch in range(epochs):
            print("epoch number " + str(epoch+1))

            self.train()
            for elem in train_loader:
                X, y = elem[0].to(self.device), elem[1].to(self.device)

                for estimator, optimizer in zip(estimators, optimizers):

                    optimizer.zero_grad()

                    if algorithm == 'fgsm' or algorithm == 'FGSM':
                       delta = self.fgsm(estimator, X, y, epsilon=epsilon) 
                    else:
                       delta = self.pgd(estimator, X, y, epsilon=epsilon, num_iter=num_iter, alpha=alpha) 

                    #elem[0] += delta # input + perturbation
                    #data, target = io.split_data_target(elem, self.device)

                    #output = estimator(*data)
                       
                    X_input = X+delta
                    output = estimator(X_input)
                    loss = self._criterion(output, y)
                    loss.backward()
                    optimizer.step()



                total_iters += 1

            if algorithm=='fgsm' or algorithm=='FGSM':
                obj_test.testModel_logs(dataset, modelName, epoch+1, 'std_fgsm', ratio ,epsilon, 0, 0, ratio_adv, time.time() - t_init, write_pred_logs, num_samples=num_samples)
            else:
                obj_test.testModel_logs(dataset, modelName, epoch+1, 'std_pgd', ratio ,epsilon, num_iter, alpha, ratio_adv, time.time() - t_init, write_pred_logs, num_samples=num_samples)


    def fgsm(self, model, X, y, epsilon=0.1):
        """ Construct FGSM adversarial examples on the examples X"""
    
        delta = torch.zeros_like(X, requires_grad=True)
        X_input = X + delta
        y_pred = model(X_input)

        loss = self._criterion(y_pred, y)
        loss.backward()

        return epsilon * delta.grad.detach().sign()
    

    def pgd(self, model, X, y, epsilon=0.1, alpha=0.01, num_iter=20, randomize=False):
        """ Construct PGD adversarial examples on the examples X"""
        if randomize:
            delta = torch.rand_like(X, requires_grad=True)
            delta.data = delta.data * 2 * epsilon - epsilon
        else:
            delta = torch.zeros_like(X, requires_grad=True)
        delta.requires_grad = True
        
        for _ in range(num_iter):
            X_input = X + delta
            y_pred = model(X_input)
            
            loss = self._criterion(y_pred, y)
            loss.backward()

            delta.data = (delta + alpha*delta.grad.detach().sign()).clamp(-epsilon,epsilon)
            delta.grad.zero_()

        return delta.detach()
    

class AdversarialTrainingClassifier(AdversarialTrainingClassifier):
    def fit(
        self,
        train_loader,
        epochs=100,
        epsilon=0.5,
        log_interval=100,
        test_loader=None,
        save_model=True,
        save_dir=None,
        obj_test=None,
        dataset=None, 
        modelName=None, 
        write_pred_logs=True, 
        num_samples=1,
        t_init=0,
        algorithm='fgsm',
        num_iter=10, 
        alpha=0.01,   
        ratio=1.0,
        ratio_adv=1.0

    ):


        self._validate_parameters(epochs, epsilon, log_interval)
        self.n_outputs = self._decide_n_outputs(train_loader)

        # Instantiate a pool of base estimators, optimizers, and schedulers.
        estimators = []
        for _ in range(self.n_estimators):
            estimators.append(self._make_estimator())

        optimizers = []
        for i in range(self.n_estimators):
            optimizers.append(
                set_module.set_optimizer(
                    estimators[i], self.optimizer_name, **self.optimizer_args
                )
            )

        if self.use_scheduler_:
            scheduler_ = set_module.set_scheduler(
                optimizers[0], self.scheduler_name, **self.scheduler_args
            )

        # Check the training criterion
        if not hasattr(self, "_criterion"):
            self._criterion = nn.CrossEntropyLoss()

        # Utils
        best_acc = 0.0

        # Internal helper function on pesudo forward
        def _forward(estimators, *x):
            outputs = [
                F.softmax(estimator(*x), dim=1) for estimator in estimators
            ]
            proba = op.average(outputs)

            return proba

        # Maintain a pool of workers
        with Parallel(n_jobs=self.n_jobs) as parallel:

            # Training loop
            for epoch in range(epochs):
                self.train()

                if self.use_scheduler_:
                    if self.scheduler_name == "ReduceLROnPlateau":
                        cur_lr = optimizers[0].param_groups[0]["lr"]
                    else:
                        cur_lr = scheduler_.get_last_lr()[0]
                else:
                    cur_lr = None

                if self.n_jobs and self.n_jobs > 1:
                    msg = "Parallelization on the training epoch: {:03d}"
                    self.logger.info(msg.format(epoch))

                rets = parallel(
                    delayed(_parallel_fit_per_epoch_adv)(
                        train_loader,
                        epsilon,
                        estimator,
                        cur_lr,
                        optimizer,
                        self._criterion,
                        #idx,
                        #epoch,
                        #log_interval,
                        self.device,
                        #True,
                        algorithm=algorithm,
                        num_iter=num_iter, 
                        alpha=alpha,  
                    )
                    for idx, (estimator, optimizer) in enumerate(
                        zip(estimators, optimizers)
                    )
                )

                estimators, optimizers, losses = [], [], []
                for estimator, optimizer, loss in rets:
                    estimators.append(estimator)
                    optimizers.append(optimizer)
                    losses.append(loss)

                self.estimators_ = nn.ModuleList()
                self.estimators_.extend(estimators)
                if save_model:
                    io.save(self, save_dir, self.logger)

                if algorithm=='fgsm' or algorithm=='FGSM':
                    test_err, _, _, _, _, _, _, _ = obj_test.testModel_logs(dataset, modelName, epoch+1, 'std_fgsm', ratio ,epsilon, 0, 0, ratio_adv, time.time() - t_init, write_pred_logs, num_samples=num_samples)
                else:
                    test_err, _, _, _, _, _, _, _ = obj_test.testModel_logs(dataset, modelName, epoch+1, 'std_pgd', ratio ,epsilon, num_iter, alpha, ratio_adv, time.time() - t_init, write_pred_logs, num_samples=num_samples)

                acc = 1 - test_err
                
                # Update the scheduler
                with warnings.catch_warnings():

                    # UserWarning raised by PyTorch is ignored because
                    # scheduler does not have a real effect on the optimizer.
                    warnings.simplefilter("ignore", UserWarning)

                    if self.use_scheduler_:
                        if self.scheduler_name == "ReduceLROnPlateau":
                            if test_loader:
                                scheduler_.step(acc)
                            else:
                                loss = torch.mean(torch.tensor(losses))
                                scheduler_.step(loss)
                        else:
                            scheduler_.step()


