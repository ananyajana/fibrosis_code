import shutil
import time
import os
import math
import json

import torch
import torch.nn as nn
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.optim
import torchvision.transforms as transforms
from models import BaselineNet, BaselineNet2
import numpy as np
from sklearn import metrics

from options import Options
from dataset import LiverDataset
import utils


def main():
    global best_score, logger, logger_results, slide_weights
    opt = Options(isTrain=True)
    opt.parse()
    opt.save_options()

    # tb_writer = SummaryWriter('{:s}/tb_logs'.format(opt.train['save_dir']))

    os.environ['CUDA_VISIBLE_DEVICES'] = ','.join(str(x) for x in opt.train['gpus'])

    # set up logger
    logger, logger_results = utils.setup_logger(opt)
    opt.print_options(logger)

    if opt.train['random_seed'] >= 0:
        # logger.info("=> Using random seed {:d}".format(opt.random_seed))
        torch.manual_seed(opt.train['random_seed'])
        torch.cuda.manual_seed(opt.train['random_seed'])
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
        np.random.seed(opt.train['random_seed'])
    else:
        torch.backends.cudnn.benchmark = True

    # ---------- Create model ---------- #
    #print('in train file', opt.model['use_resnet'])
    if opt.model['use_resnet'] == 1:
        #print('opt model use_resnet ', type(opt.model['use_resnet']))
        #print('choosing baseline2')
        model = BaselineNet2(opt.model['out_c'], opt.model['resnet_layers'], opt.model['train_res4'])
        logger.info('choosing BaselineNet2(resnet based) model')
    else:
        model = BaselineNet(opt.model['in_c'], opt.model['out_c'])
        logger.info('choosing BaselineNet model')
        if opt.model['pre_train_sup'] == 1:
            print("=> loading pre-trained model from path", opt.model['sup_model_path'])
            model.set_fea_extractor(opt.model['sup_model_path'])

    model = model.cuda()
    # logger.info(model)
    # ---------- End create model ---------- #

    # define loss function (criterion) and optimizer
    criterion = nn.CrossEntropyLoss().cuda()
    optimizer = torch.optim.Adam(model.parameters(), opt.train['lr'], weight_decay=opt.train['weight_decay'])

    # ---------- Data loading ---------- #
    data_transform = transforms.Compose([
                    transforms.Resize((224, 224)),
                    transforms.ToTensor(),
                    transforms.Normalize(mean=[0.485],
                                         std=[0.229])
                ])

    fold_num = opt.exp_num.split('_')[-1]
    logger.info('Fold number: {:s}'.format(fold_num))
    if opt.model['use_resnet'] == 1:
        train_set = LiverDataset('{:s}/train{:s}.h5'.format(opt.train['data_dir'], fold_num), data_transform, opt)
        test_set = LiverDataset('{:s}/test{:s}.h5'.format(opt.train['data_dir'], fold_num), data_transform, opt)
    else:
        train_set = LiverDataset('{:s}/train{:s}.h5'.format(opt.train['data_dir'], fold_num), data_transform)
        test_set = LiverDataset('{:s}/test{:s}.h5'.format(opt.train['data_dir'], fold_num), data_transform)
    # loading from the pre-trained model
    #print("=> loading pre-trained model from path", opt.model['sup_model_path'])
    #model.set_fea_extractor(opt.model['sup_model_path'])
    # ---------- End Data loading ---------- #

    # ----- optionally load from a checkpoint ----- #
    # if opt.train['checkpoint']:
    #     model_state_dict, optimizer_state_dict = load_checkpoint(opt.train['checkpoint'])
    #     model.load_state_dict(model_state_dict)
    #     optimizer.load_state_dict(optimizer_state_dict)
    # ----- End checkpoint loading ----- #

    # ----- Start training ---- #
    best_score = 0
    for epoch in range(opt.train['epochs']):
        # train and validate for one epoch
        train_loss, train_acc = train(opt, train_set, model, criterion, optimizer, epoch)
        test_loss, test_acc, test_auc = test(opt, test_set, model, criterion, epoch)

        # remember best accuracy and save checkpoint
        is_best = test_auc > best_score
        best_score = max(test_auc, best_score)
        cp_flag = False
        save_checkpoint({
            'epoch': epoch + 1,
            'state_dict': model.state_dict(),
            'best_score': best_score,
            'optimizer': optimizer.state_dict(),
        }, is_best, opt.train['save_dir'], cp_flag, epoch+1)

        # save training results
        logger_results.info('{:<6d}| {:<12.4f}{:<12.4f}||  {:<12.4f}{:<12.4f}{:<12.4f}'
                            .format(epoch, train_loss, train_acc,
                                    test_loss, test_acc, test_auc))

    for i in list(logger.handlers):
        logger.removeHandler(i)
        i.flush()
        i.close()
    for i in list(logger_results.handlers):
        logger_results.removeHandler(i)
        i.flush()
        i.close()


def train(opt, train_set, model, criterion, optimizer, epoch):
    batch_time = utils.AverageMeter()
    data_time = utils.AverageMeter()
    losses = utils.AverageMeter()
    acc = utils.AverageMeter()

    # switch to train mode
    model.train()

    end = time.time()
    idx_list = torch.randperm(len(train_set)).tolist()
    # idx_list = torch.multinomial(torch.Tensor(slide_weights), len(train_set), replacement=True)

    for i in range(0, len(idx_list), opt.train['batch_size']):
        N = min(opt.train['batch_size'], len(idx_list) - i)
        loss = 0
        for k in range(N):
            idx = idx_list[i + k]
            ct_data, fib_label, nas_stea_label, nas_lob_label, nas_balloon_label = train_set[idx]
            if opt.exp == 'fib':
                label = fib_label
            elif opt.exp == 'nas_stea':
                label = nas_stea_label
            elif opt.exp == 'nas_lob':
                label = nas_lob_label
            elif opt.exp == 'nas_balloon':
                label = nas_balloon_label
            else:
                raise ValueError('Wrong label name')

            input_data = ct_data

            output = model(input_data.cuda())
            #raise ValueError('Exit')
            loss += criterion(output, label.cuda())

            # measure accuracy
            probs = nn.functional.softmax(output, dim=1)
            pred = torch.argmax(probs, dim=1).cpu()
            accuracy = (pred == label).sum().numpy()

            acc.update(accuracy)

            del output

        loss /= N
        losses.update(loss.item(), N)

        # compute gradient and do SGD step
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

        if i % opt.train['log_interval'] == 0:
            logger.info('Epoch: [{0}][{1}/{2}]\t'
                        'Data Time: {data_time.avg:.3f}\t'
                        'Batch Time: {batch_time.avg:.3f}\t'
                        'Loss: {loss.avg:.3f}\t'
                        'Acc: {acc.avg:.4f}'
                        .format(epoch, i, len(train_set), data_time=data_time, batch_time=batch_time,
                                loss=losses, acc=acc))

    logger.info('=> Train Avg: Loss: {loss.avg:.3f}\t\tAcc: {acc.avg:.4f}'
                .format(loss=losses, acc=acc))

    return losses.avg, acc.avg


def test(opt, test_set, model, criterion, epoch):
    batch_time = utils.AverageMeter()
    losses = utils.AverageMeter()
    acc = utils.AverageMeter()

    # switch to evaluate mode
    model.eval()

    end = time.time()
    slide_probs_all = []
    slide_targets_all = []
    for i in range(len(test_set)):
        ct_data, fib_label, nas_stea_label, nas_lob_label, nas_balloon_label = test_set[i]
        if opt.exp == 'fib':
            label = fib_label
        elif opt.exp == 'nas_stea':
            label = nas_stea_label
        elif opt.exp == 'nas_lob':
            label = nas_lob_label
        elif opt.exp == 'nas_balloon':
            label = nas_balloon_label
        else:
            raise ValueError('Wrong label name')

        with torch.no_grad():
            input_data = ct_data
            output = model(input_data.cuda())
            loss = criterion(output, label.cuda())

        # measure accuracy and record loss
        probs = nn.functional.softmax(output, dim=1)
        pred = torch.argmax(probs, dim=1).detach().cpu()
        accuracy = (pred == label).sum().detach().cpu().numpy()

        # measure accuracy and record loss
        losses.update(loss.item())
        acc.update(accuracy)

        slide_probs_all.append(probs.detach().cpu())
        slide_targets_all.append(label.detach().cpu())

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

        del output, probs

        if i % opt.train['log_interval'] == 0:
            logger.info('Test: [{0}][{1}/{2}]\t'
                        'Time: {batch_time.avg:.3f}\t'
                        'Loss: {loss.avg:.3f}'
                        .format(epoch, i, len(test_set), batch_time=batch_time, loss=losses))

    slide_probs_all = torch.cat(slide_probs_all, dim=0).numpy()
    slide_targets_all = torch.cat(slide_targets_all, dim=0).numpy()

    pred = np.argmax(slide_probs_all, axis=1)
    acc = metrics.accuracy_score(slide_targets_all, pred)

    if opt.exp in ['fib', 'nas_lob', 'nas_balloon']:
        # Compute ROC curve and ROC area for each class
        fpr = dict()
        tpr = dict()
        roc_auc = dict()
        for i in range(3):
            fpr[i], tpr[i], _ = metrics.roc_curve(slide_targets_all == i, slide_probs_all[:, i])
            auc_i = metrics.auc(fpr[i], tpr[i])
            roc_auc[i] = 0 if math.isnan(auc_i) else auc_i
        auc = np.mean(np.array(list(roc_auc.values())))

        logger.info('Test Avg: {}\tLoss: {:.3f}\tAcc: {:.4f}\tAUC: {:.4f}\n'
                    'AUC0: {:.4f}\tAUC1: {:.4f}\tAUC2: {:.4f}\n'
                    .format(epoch, losses.avg, acc, auc, roc_auc[0], roc_auc[1], roc_auc[2]))
    else:
        tp = np.sum((pred == 1) * (slide_targets_all == 1))
        tn = np.sum((pred == 0) * (slide_targets_all == 0))
        fp = np.sum((pred == 1) * (slide_targets_all == 0))
        fn = np.sum((pred == 0) * (slide_targets_all == 1))
        acc = metrics.accuracy_score(slide_targets_all, pred)
        auc = metrics.roc_auc_score(slide_targets_all, slide_probs_all[:, 1])

        logger.info('Test Avg: {}\tLoss: {:.3f}\tAcc: {:.4f}\tAUC: {:.4f}\n'
                    'TP: {:d}\tTN: {:d}\tFP: {:d}\tFN: {:d}\n'
                    .format(epoch, losses.avg, acc, auc, tp, tn, fp, fn))

    return losses.avg, acc, auc


def save_checkpoint(state, is_best, save_dir, cp_flag, epoch):
    if not os.path.exists(save_dir):
        #os.mkdir(save_dir)
        os.makedirs(save_dir, exist_ok=True)
    if is_best:
        print('saving mode at {}'.format(save_dir))
        torch.save(state, '{:s}/checkpoint_best.pth.tar'.format(save_dir))
    # filename = '{:s}/checkpoint.pth.tar'.format(save_dir)
    # torch.save(state, filename)
    # if cp_flag:
    #     shutil.copyfile(filename, '{:s}/checkpoint_{:d}.pth.tar'.format(save_dir, epoch))
    # if is_best:
    #     shutil.copyfile(filename, '{:s}/checkpoint_best.pth.tar'.format(save_dir))


def load_checkpoint(checkpoint_path):
    model_state_dict = None
    optimizer_state_dict = None
    if os.path.isfile(checkpoint_path):
        logger.info("=> loading checkpoint '{}'".format(checkpoint_path))
        checkpoint = torch.load(checkpoint_path)
        model_state_dict = checkpoint['state_dict']
        optimizer_state_dict = checkpoint['optimizer']
        logger.info("=> loaded checkpoint '{}' (epoch {})".format(checkpoint_path, checkpoint['epoch']))
    else:
        logger.info("=> no checkpoint found at '{}'".format(checkpoint_path))
    return model_state_dict, optimizer_state_dict


def assign_slide_weight(bags, weights):
    slide_weights = np.zeros(len(bags))
    for i in range(len(bags)):
        bag = bags[i]
        target = bag['label']
        slide_weights[i] = weights[0] if target == 0 else weights[1]
    return slide_weights


if __name__ == '__main__':
    main()
