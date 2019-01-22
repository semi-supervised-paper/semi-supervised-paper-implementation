import os
import time
import re

import torch

from .basic_method import BasicMethod
from ..utils.loss_util import VATLoss
from ..utils.log_util import AverageMeter, AverageMeterSet, GenericCSV
from ..utils.fun_util import accuracy, save_checkpoint_to_file, save_best_checkpoint_to_file, parameters_string
from ..utils.constant import DATA_NO_LABEL, METHOD_VAT


class VAT(BasicMethod):
    def __init__(self,
                 train_loader,
                 eval_loader,
                 num_classes,
                 args):
        super(VAT, self).__init__(train_loader, eval_loader, num_classes, args)
        self.model = self.create_model(ema=False, args=args)
        print(parameters_string(self.model))

        self.optimizer = torch.optim.Adam(self.model.parameters(), self.args.lr, betas=(0.9, 0.999), eps=1e-8)

        self.loss_ce = torch.nn.CrossEntropyLoss(ignore_index=DATA_NO_LABEL).cuda()
        self.loss_vat = VATLoss()

        self.map = dict.fromkeys(['Epoch', 'EpochTime', 'TrainLoss', 'TestLoss', 'TestAccuracy',
                                  'TrainSupervisedLoss', 'TrainConsisencyLoss', 'TrainUnsupervisedLoss',
                                  'LearningRate'])

        n_labels = re.findall(r"(\d+)_balanced_labels", str(args.labels))
        if len(n_labels) == 0:
            labels_str = "all_labels"
        else:
            labels_str = str(n_labels[0])
        self.training_csv = GenericCSV(os.path.join(self.result_folder, 'training_label_{lb}_seed_{se}.csv'
                                                    .format(lb=labels_str, se=args.seed)),
                                       *list(self.map.keys()))

    def adjust_optimizer_params(self, optimizer, epoch):
        if epoch < self.args.epoch_decay_start:
            for param_group in optimizer.param_groups:
                super(VAT, self).log_to_tf("lr", param_group['lr'], epoch, True)
                super(VAT, self).log_to_tf("beta1", param_group['betas'][0], epoch, True)
                return param_group['lr']
        else:
            decayed_lr = ((self.args.epochs - epoch) / float(
                        self.args.epochs - self.args.epoch_decay_start)) * self.args.lr
            for param_group in optimizer.param_groups:
                param_group['lr'] = decayed_lr
                _, old_beta2 = param_group['betas']
                param_group['betas'] = self.args.beta1_2, old_beta2

            super(VAT, self).log_to_tf("lr", decayed_lr, epoch, True)
            super(VAT, self).log_to_tf("beta1", self.args.beta1_2, epoch, True)
            return decayed_lr

    def train_model(self):
        for epoch in range(self.start_epoch, self.args.epochs):
            print("epoch {e}".format(e=epoch))
            start = time.time()
            
            self._train_one_epoch(epoch)
            top1_avg_validate, top5_avg_validate, class_loss_avg_validate = self._validate(epoch)

            if self.best_top1_validate is None or self.best_top1_validate < top1_avg_validate:
                self.best_top1_validate = top1_avg_validate
                self.best_top5_validate = top5_avg_validate
                is_best = True
            else:
                is_best = False

            self._save_checkpoint(epoch, self.global_step, top1_avg_validate, top5_avg_validate,
                                  self.best_top1_validate, self.best_top5_validate,
                                  class_loss_avg_validate, is_best)

            self.training_csv.add_data(*list(self.map.values()))
            end = time.time()
            print("EPOCH {e} use {time} s".format(e=epoch, time=(end-start)))
        print("best test top1 accuracy is {acc}".format(acc=self.best_top1_validate))

    @staticmethod
    def get_params():
        return {
            'method': METHOD_VAT,
            'epoch_decay_start': 80,
            'beta1_2': 0.5,
            'vat_wt': 1.
        }

    def _train_one_epoch(self, epoch):
        batch_time = AverageMeter()
        data_time = AverageMeter()
        losses_ce = AverageMeter()
        losses_vat = AverageMeter()
        top1 = AverageMeter()
        top5 = AverageMeter()
        losses_all = AverageMeter()

        start = time.time()
        end = time.time()

        lr = self.adjust_optimizer_params(self.optimizer, epoch)

        self.model.train()

        total_data_size, total_labeled_size = 0, 0
        for i, (input, target) in enumerate(self.train_loader):
            if isinstance(input, tuple) or isinstance(input, list):
                input_var = torch.autograd.Variable(input[0])
            else:
                input_var = torch.autograd.Variable(input)
            try:
                target_var = torch.autograd.Variable(target.cuda(async=True))
            except:
                target_var = torch.autograd.Variable(target)
                
            data_time.update(time.time() - end)
            minibatch_size = len(target_var)
            labeled_minibatch_size = target_var.data.ne(DATA_NO_LABEL).sum().float()
            total_data_size += minibatch_size
            total_labeled_size += labeled_minibatch_size

            input_labeled_index = (target != DATA_NO_LABEL).view(-1).nonzero().view(-1)
            input_unlabeled_index = (target == DATA_NO_LABEL).view(-1).nonzero().view(-1)

            if input_labeled_index is None or len(input_labeled_index) == 0:
                loss_ce = torch.cuda.FloatTensor([0])
                output_1 = None
            else:
                output_1 = self.model(input_var[input_labeled_index])
                loss_ce = self.loss_ce(output_1, target_var[input_labeled_index])

            loss_vat = self.args.vat_wt * self.loss_vat(self.model, input_var[input_unlabeled_index])

            if i % 50 == 0:
                print("cur labeled_size is {ls}, cur minibatch_size is {ms}, loss_ce = {ce}, weighted_loss_vat = {vat}"
                      .format(ls=labeled_minibatch_size, ms=minibatch_size,ce=loss_ce, vat=loss_vat.item()))

            loss = loss_ce + loss_vat
            losses_all.update(loss.item())

            super(VAT, self).log_to_tf("supervised_loss", loss_ce.item(), self.global_step, True)
            super(VAT, self).log_to_tf("consistency_loss", loss_vat.item(), self.global_step, True)
            # measure accuracy and record loss
            if self.args.topk == 1 and output_1 is not None:
                prec1 = accuracy(output_1.data, target_var[input_labeled_index].data, topk=(1, ))[0]
                top1.update(prec1.item(), labeled_minibatch_size)
            elif output_1 is not None:
                prec1, prec5 = accuracy(output_1.data, target_var[input_labeled_index].data, topk=(1, 5))
                top5.update(prec5.item(), labeled_minibatch_size)
                top1.update(prec1.item(), labeled_minibatch_size)

            losses_ce.update(loss_ce.item())
            losses_vat.update(loss_vat.item())

            # compute gradient and do SGD step
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

            # measure elapsed time
            batch_time.update(time.time() - end)
            end = time.time()
            self.global_step += 1

        print("labeled_minibatch_size is {lms}, total_size is {ts}".format(lms=total_labeled_size, ts=total_data_size))
        print('Time {time}, Epoch: {e}: Loss_CE_Epoch {Loss_CE_Epoch}, Loss_Consisency_Epoch {Loss_Consisency_Epoch},'
              'Loss_All_Epoch: {Loss_All_Epoch}, Train_Top1_Epoch: {Train_Top1_Epoch},'
              'Train_Top5_Epoch: {Train_Top5_Epoch}, Train_Error1_Epoch: {Train_Error1_Epoch}, '
              'Train_Error5_Epoch: {Train_Error5_Epoch}, Learning_Rate: {lr}'.format(
                time=time.time()-start,
                e=epoch, Loss_CE_Epoch=losses_ce.avg, Loss_Consisency_Epoch=losses_vat.avg,
                Loss_All_Epoch=losses_all.avg, Train_Top1_Epoch=top1.avg, Train_Top5_Epoch=top5.avg,
                Train_Error1_Epoch= 100.0 - top1.avg, Train_Error5_Epoch=100.0 - top5.avg,
                lr=lr
              ))

        super(VAT, self).log_to_tf("Loss_CE_Epoch", losses_ce.avg, epoch, True)
        super(VAT, self).log_to_tf("Loss_Consisency_Epoch", losses_vat.avg, epoch, True)
        super(VAT, self).log_to_tf("Loss_All_Epoch", losses_all.avg, epoch, True)
        super(VAT, self).log_to_tf("Train_Top1_Epoch", top1.avg, epoch, True)
        super(VAT, self).log_to_tf("Train_Top5_Epoch", top5.avg, epoch, True)
        super(VAT, self).log_to_tf("Train_Error1_Epoch", 100.0 - top1.avg, epoch, True)
        super(VAT, self).log_to_tf("Train_Error5_Epoch", 100.0 - top5.avg, epoch, True)

        self.map['Epoch'] = epoch
        self.map['EpochTime'] = time.time() - start
        self.map['TrainLoss'] = losses_all.avg
        self.map['TrainConsisencyLoss'] = losses_vat.avg
        self.map['TrainSupervisedLoss'] = losses_ce.avg
        self.map['TrainUnsupervisedLoss'] = losses_vat.avg
        self.map['LearningRate'] = lr

    def _validate(self, epoch=None):
        class_criterion = torch.nn.CrossEntropyLoss(reduction="sum", ignore_index=DATA_NO_LABEL).cuda()
        meters = AverageMeterSet()

        # switch to evaluate mode
        self.model.eval()

        end = time.time()

        with torch.no_grad():
            for i, (input, target) in enumerate(self.eval_loader):
                meters.update('data_time', time.time() - end)

                input_var = torch.autograd.Variable(input)
                try:
                    target_var = torch.autograd.Variable(target.cuda(async=True))
                except:
                    target_var = torch.autograd.Variable(target)

                minibatch_size = len(target_var)
                labeled_minibatch_size = target_var.data.ne(DATA_NO_LABEL).sum().float()
                assert labeled_minibatch_size and minibatch_size == labeled_minibatch_size
                meters.update('labeled_minibatch_size', labeled_minibatch_size.item())

                # compute output
                output = self.model(input_var)
                class_loss = class_criterion(output, target_var) / minibatch_size

                # measure accuracy and record loss
                if self.args.topk == 1:
                    prec1 = accuracy(output.data, target_var.data, topk=(1, ))[0]
                    meters.update('top5', 0, labeled_minibatch_size.item())
                    meters.update('error5', 100.0, labeled_minibatch_size.item())
                else:
                    prec1, prec5 = accuracy(output.data, target_var.data, topk=(1, 5))
                    meters.update('top5', prec5[0], labeled_minibatch_size.item())
                    meters.update('error5', 100.0 - prec5[0], labeled_minibatch_size.item())
                meters.update('class_loss', class_loss.item(), labeled_minibatch_size.item())
                meters.update('top1', prec1[0], labeled_minibatch_size.item())
                meters.update('error1', 100.0 - prec1[0], labeled_minibatch_size.item())

                # measure elapsed time
                meters.update('batch_time', time.time() - end)
                end = time.time()

            if epoch is not None:
                super(VAT, self).log_to_tf("Test_Top1_Epoch", meters['top1'].avg, epoch, False)
                super(VAT, self).log_to_tf("Test_Top5_Epoch", meters['top5'].avg, epoch, False)
                super(VAT, self).log_to_tf("Test_Error1_Epoch", 100.0 - meters['top1'].avg, epoch, False)
                super(VAT, self).log_to_tf("Test_Error5_Epoch", 100.0 - meters['top5'].avg, epoch, False)
                super(VAT, self).log_to_tf("Test_Class_Loss", meters['class_loss'].avg, epoch, False)

                self.map['TestLoss'] = meters['class_loss'].avg
                self.map['TestAccuracy'] = meters['top1'].avg

            print(' * Prec@1 {top1.avg:.3f}\tPrec@5 {top5.avg:.3f}\tClass_Loss {cl.avg:.5f}'
                  .format(top1=meters['top1'], top5=meters['top5'], cl=meters['class_loss']))

        return meters['top1'].avg, meters['top5'].avg, meters['class_loss'].avg

    def _save_checkpoint(self,
                          epoch, global_step, top1_validate, top5_validate,
                          best_top1_validate, best_top5_validate,
                          class_loss_validate, is_best):
        if is_best:
            save_best_checkpoint_to_file({
                'epoch': epoch,
                'global_step': global_step,
                'semi-supervised-method': 'VAT',
                'state_dict': self.model.state_dict(),
                'optimizer': self.optimizer.state_dict(),
                'top1_validate': top1_validate,
                'top5_validate': top5_validate,
                'best_top1_validate': best_top1_validate,
                'best_top5_validate': best_top5_validate,
                'class_loss_validate': class_loss_validate
            }, is_best, self.result_folder)

        if epoch % self.args.checkpoint_epochs == 0:
            save_checkpoint_to_file({
                'epoch': epoch,
                'global_step': global_step, 
                'semi-supervised-method': 'VAT',
                'state_dict': self.model.state_dict(),
                'optimizer': self.optimizer.state_dict(),
                'top1_validate': top1_validate,
                'top5_validate': top5_validate,
                'best_top1_validate': best_top1_validate,
                'best_top5_validate': best_top5_validate,
                'class_loss_validate': class_loss_validate
            }, epoch, is_best, self.result_folder)

    def _load_checkpoint(self, filepath):
        if os.path.isfile(filepath):
            print("=> loading checkpoint '{}'".format(filepath))
            checkpoint = torch.load(filepath)
            self.start_epoch = checkpoint['epoch'] + 1
            self.global_step = checkpoint['global_step'] + 1
            self.best_top1_validate = checkpoint['best_top1_validate']
            self.best_top5_validate = checkpoint['best_top5_validate']
            self.model.load_state_dict(checkpoint['state_dict'])
            self.optimizer.load_state_dict(checkpoint['optimizer'])
            print("=> loaded checkpoint '{}' (epoch {}) "
                  "best_top1_validate = {}, best_top5_validate = {}, "
                  "top1_validate = {}, top5_validate = {}, class_loss_validate = {}"
                  .format(filepath, checkpoint['epoch'], self.best_top1_validate, self.best_top5_validate,
                          checkpoint['top1_validate'], checkpoint['top5_validate'], checkpoint['class_loss_validate']))
        else:
            print("=> no checkpoint found at '{}'".format(filepath))
