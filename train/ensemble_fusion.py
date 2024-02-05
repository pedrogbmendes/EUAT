import torch
import torch.nn as nn
import torch.nn.functional as F

from torchensemble import FusionClassifier
from torchensemble.utils import io
from torchensemble.utils import set_module
from torchensemble.utils import operator as op
import time

class FusionClassifier(FusionClassifier):

    def fit(
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
            # # Validation
            # if test_loader:
            #     self.eval()

        #         with torch.no_grad():
        #             correct = 0
        #             total = 0
        #             for _, elem in enumerate(test_loader):
        #                 data, target = io.split_data_target(elem, self.device)
        #                 output = self.forward(*data)
        #                 _, predicted = torch.max(output.data, 1)
        #                 correct += (predicted == target).sum().item()
        #                 total += target.size(0)
        #             acc = 100 * correct / total

        #             if acc > best_acc:
        #                 best_acc = acc
        #                 if save_model:
        #                     io.save(self, save_dir, self.logger)

        #             msg = (
        #                 "Epoch: {:03d} | Validation Acc: {:.3f}"
        #                 " % | Historical Best: {:.3f} %"
        #             )
        #             self.logger.info(msg.format(epoch, acc, best_acc))
        #             if self.tb_logger:
        #                 self.tb_logger.add_scalar(
        #                     "fusion/Validation_Acc", acc, epoch
        #                 )

        #     # Update the scheduler
        #     if hasattr(self, "scheduler_"):
        #         if self.scheduler_name == "ReduceLROnPlateau":
        #             if test_loader:
        #                 self.scheduler_.step(acc)
        #             else:
        #                 self.scheduler_.step(loss)
        #         else:
        #             self.scheduler_.step()

        # if save_model and not test_loader:
        #     io.save(self, save_dir, self.logger)