import os
import time

import torch

from .basic_method import BasicMethod
from ..utils import fun_util
from ..utils import loss_util
from ..utils.log_util import AverageMeter, AverageMeterSet, GenericCSV
from ..utils.data_util import ZCATransformation
from ..utils.constant import DATA_NO_LABEL, METHOD_TEMPORAL_ENSEMBLING


class TemporalEnsembling(BasicMethod):
    def __init__(self,
                 train_loader,
                 eval_loader,
                 num_classes,
                 args):
        super(TemporalEnsembling, self).__init__(train_loader, eval_loader, num_classes, args)

        self.loss_ce = torch.nn.CrossEntropyLoss(reduction='sum', ignore_index=DATA_NO_LABEL).cuda()
        self.loss_pi = loss_util.softmax_mse_loss

        self.map = dict.fromkeys(['Epoch', 'EpochTime', 'TrainLoss', 'TestLoss', 'TestAccuracy',
                                  'TrainSupervisedLoss', 'TrainPiLoss', 'TrainUnsupervisedLoss',
                                  'LearningRate'])

        self.training_csv = GenericCSV(os.path.join(self.result_folder, 'training_label_{lb}_seed_{se}.csv'
                                                    .format(lb=self.labels_str, se=args.seed)),
                                       *list(self.map.keys()))

    def adjust_optimizer_params(self, optimizer, epoch):
        rampup_value = fun_util.rampup(epoch, self.args.rampup_epoch)
        rampdown_value = fun_util.rampdown(epoch, self.args.epochs, self.args.rampdown_epoch)
        learning_rate = rampup_value * rampdown_value * self.args.lr
        for param_group in optimizer.param_groups:
            param_group['lr'] = learning_rate
            old_beta1, old_beta2 = param_group['betas']
            adam_beta1 = rampdown_value * 0.9 + (1.0 - rampdown_value) * 0.5
            param_group['betas'] = adam_beta1, old_beta2

        super(TemporalEnsembling, self).log_to_tf("lr", learning_rate, epoch, True)
        super(TemporalEnsembling, self).log_to_tf("beta1", adam_beta1, epoch, True)
        return learning_rate

    def adjust_consistency_weight(self, epoch):
        return self.args.consistency * fun_util.rampup(epoch, self.args.rampup_epoch)

    def train_model(self):
        for epoch in range(self.start_epoch, self.args.epochs):
            print("epoch {e}".format(e=epoch))
            start = time.time()
            if epoch == self.start_epoch:
                cur_cons_weight = 0
            else:
                cur_cons_weight = self.adjust_consistency_weight(epoch)

            if epoch == 0:
                self._train_one_epoch(epoch, cur_cons_weight, init_mode=True)

            self._train_one_epoch(epoch, cur_cons_weight)
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
            print("EPOCH {e} use {time} s".format(e=epoch, time=(end - start)))
        print("best test top1 accuracy is {acc}".format(acc=self.best_top1_validate))
        self.training_csv.close()

    @staticmethod
    def get_params():
        return {
            'method': METHOD_TEMPORAL_ENSEMBLING
        }

    def _train_one_epoch(self, epoch, cons_weight, init_mode=False):
        if not init_mode:
            batch_time = AverageMeter()
            data_time = AverageMeter()
            losses_ce = AverageMeter()
            losses_pi = AverageMeter()
            top1 = AverageMeter()
            top5 = AverageMeter()
            losses_all = AverageMeter()

            start = time.time()
            end = time.time()

            lr = self.adjust_optimizer_params(self.optimizer, epoch)
            super(TemporalEnsembling, self).log_to_tf("cons_weight", cons_weight, epoch, True)

        self.model.train()
        total_data_size, total_labeled_size = 0, 0
        for i, (input_pack, target) in enumerate(self.train_loader):
            (input, input2) = input_pack
            input = self.zca(input)
            input2 = self.zca(input2)

            input_1_var = torch.autograd.Variable(input)
            input_2_var = torch.autograd.Variable(input2)
            try:
                target_var = torch.autograd.Variable(target.cuda(async=True))
            except:
                target_var = torch.autograd.Variable(target)

            if init_mode:
                with torch.no_grad():
                    self.model(input_1_var, moving_average=True, init_mode=True)
                break

            data_time.update(time.time() - end)
            minibatch_size = len(target_var)
            labeled_minibatch_size = target_var.data.ne(DATA_NO_LABEL).sum().float()
            total_data_size += minibatch_size
            total_labeled_size += labeled_minibatch_size

            output_1 = self.model(input_1_var, moving_average=True)
            output_2 = self.model(input_2_var, moving_average=False)

            loss_ce = self.loss_ce(output_1, target_var) / minibatch_size
            loss_pi = cons_weight * self.loss_pi(output_1, output_2) / minibatch_size

            loss = loss_ce + loss_pi
            losses_all.update(loss.item())
            super(TemporalEnsembling, self).log_to_tf("supervised_loss", loss_ce.item(), self.global_step, True)
            super(TemporalEnsembling, self).log_to_tf("consistency_loss", loss_pi.item(), self.global_step, True)

            if self.args.topk == 1:
                prec1 = fun_util.accuracy(output_1.data, target_var.data, topk=(1,))[0]
            else:
                prec1, prec5 = fun_util.accuracy(output_1.data, target_var.data, topk=(1, 5))
                top5.update(prec5.item(), labeled_minibatch_size)

            losses_ce.update(loss_ce.item())
            losses_pi.update(loss_pi.item())
            top1.update(prec1.item(), labeled_minibatch_size)

            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

            batch_time.update(time.time() - end)
            end = time.time()
            self.global_step += 1

        if not init_mode:
            print("labeled_minibatch_size is {lms}, total_size is {ts}".format(lms=total_labeled_size,
                                                                               ts=total_data_size))
            print('Time {time}, Epoch: {e}: Loss_CE_Epoch {Loss_CE_Epoch}, Loss_Consisency_Epoch'
                  '{Loss_Consisency_Epoch},'
                  'Loss_All_Epoch: {Loss_All_Epoch}, Train_Top1_Epoch: {Train_Top1_Epoch},'
                  'Train_Top5_Epoch: {Train_Top5_Epoch}, Train_Error1_Epoch: {Train_Error1_Epoch}, '
                  'Train_Error5_Epoch: {Train_Error5_Epoch}, Learning_Rate: {lr}'.format(
                    time=time.time() - start,
                    e=epoch, Loss_CE_Epoch=losses_ce.avg, Loss_Consisency_Epoch=losses_pi.avg,
                    Loss_All_Epoch=losses_all.avg, Train_Top1_Epoch=top1.avg, Train_Top5_Epoch=top5.avg,
                    Train_Error1_Epoch=100.0 - top1.avg, Train_Error5_Epoch=100.0 - top5.avg,
                    lr=lr
            ))
            super(TemporalEnsembling, self).log_to_tf("Loss_CE_Epoch", losses_ce.avg, epoch, True)
            super(TemporalEnsembling, self).log_to_tf("Loss_Consisency_Epoch", losses_pi.avg, epoch, True)
            super(TemporalEnsembling, self).log_to_tf("Loss_All_Epoch", losses_all.avg, epoch, True)
            super(TemporalEnsembling, self).log_to_tf("Train_Top1_Epoch", top1.avg, epoch, True)
            super(TemporalEnsembling, self).log_to_tf("Train_Top5_Epoch", top5.avg, epoch, True)
            super(TemporalEnsembling, self).log_to_tf("Train_Error1_Epoch", 100.0 - top1.avg, epoch, True)
            super(TemporalEnsembling, self).log_to_tf("Train_Error5_Epoch", 100.0 - top5.avg, epoch, True)

            self.map['Epoch'] = epoch
            self.map['EpochTime'] = time.time() - start
            self.map['TrainLoss'] = losses_all.avg
            self.map['TrainPiLoss'] = losses_pi.avg
            self.map['TrainSupervisedLoss'] = losses_ce.avg
            self.map['TrainUnsupervisedLoss'] = losses_pi.avg
            self.map['LearningRate'] = lr
        else:
            print("init mode finished!")

    def _validate(self, epoch=None):
        class_criterion = torch.nn.CrossEntropyLoss(reduction="sum", ignore_index=DATA_NO_LABEL).cuda()
        meters = AverageMeterSet()

        # switch to evaluate mode
        self.model.eval()

        end = time.time()

        with torch.no_grad():
            for i, (input, target) in enumerate(self.eval_loader):
                meters.update('data_time', time.time() - end)

                input = self.zca(input)
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
                    prec1 = fun_util.accuracy(output.data, target_var.data, topk=(1,))[0]
                    meters.update('top5', 0, labeled_minibatch_size.item())
                    meters.update('error5', 100.0, labeled_minibatch_size.item())
                else:
                    prec1, prec5 = fun_util.accuracy(output.data, target_var.data, topk=(1, 5))
                    meters.update('top5', prec5[0], labeled_minibatch_size.item())
                    meters.update('error5', 100.0 - prec5[0], labeled_minibatch_size.item())
                meters.update('class_loss', class_loss.item(), labeled_minibatch_size.item())
                meters.update('top1', prec1[0], labeled_minibatch_size.item())
                meters.update('error1', 100.0 - prec1[0], labeled_minibatch_size.item())

                # measure elapsed time
                meters.update('batch_time', time.time() - end)
                end = time.time()

            if epoch is not None:
                super(TemporalEnsembling, self).log_to_tf("Test_Top1_Epoch", meters['top1'].avg, epoch, False)
                super(TemporalEnsembling, self).log_to_tf("Test_Top5_Epoch", meters['top5'].avg, epoch, False)
                super(TemporalEnsembling, self).log_to_tf("Test_Error1_Epoch", 100.0 - meters['top1'].avg, epoch, False)
                super(TemporalEnsembling, self).log_to_tf("Test_Error5_Epoch", 100.0 - meters['top5'].avg, epoch, False)
                super(TemporalEnsembling, self).log_to_tf("Test_Class_Loss", meters['class_loss'].avg, epoch, False)

                self.map['TestLoss'] = meters['class_loss'].avg
                self.map['TestAccuracy'] = meters['top1'].avg

            print(' * Prec@1 {top1.avg:.3f}\tPrec@5 {top5.avg:.3f}\tClass_Loss {cl.avg:.5f}'
                  .format(top1=meters['top1'], top5=meters['top5'], cl=meters['class_loss']))

        return meters['top1'].avg, meters['top5'].avg, meters['class_loss'].avg

    def _save_checkpoint(self,
                         epoch, global_step, top1_validate, top5_validate,
                         best_top1_validate, best_top5_validate,
                         class_loss_validate, is_best):
        if not self.args.tflog:
            return

        if is_best:
            fun_util.save_best_checkpoint_to_file({
                'epoch': epoch,
                'global_step': global_step,
                'semi-supervised-method': self.args.method,
                'state_dict': self.model.state_dict(),
                'optimizer': self.optimizer.state_dict(),
                'top1_validate': top1_validate,
                'top5_validate': top5_validate,
                'best_top1_validate': best_top1_validate,
                'best_top5_validate': best_top5_validate,
                'class_loss_validate': class_loss_validate
            }, is_best, self.result_folder)

        if epoch % self.args.checkpoint_epochs == 0:
            fun_util.save_checkpoint_to_file({
                'epoch': epoch,
                'global_step': global_step,
                'semi-supervised-method': self.args.method,
                'state_dict': self.model.state_dict(),
                'optimizer': self.optimizer.state_dict(),
                'top1_validate': top1_validate,
                'top5_validate': top5_validate,
                'best_top1_validate': best_top1_validate,
                'best_top5_validate': best_top5_validate,
                'class_loss_validate': class_loss_validate
            }, epoch, is_best, self.result_folder)

    def _load_checkpoint(self, filepath):
        if not self.args.tflog:
            return

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
