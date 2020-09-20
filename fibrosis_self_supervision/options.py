import os
import numpy as np
import argparse
import torchvision.transforms as transforms


class Options:
    def __init__(self, isTrain):
        self.isTrain = isTrain  # train or test mode
        self.exp = 'fib'
        self.exp_num = 'baseline_1'

        # --- model hyper-parameters --- #
        self.model = dict()
        self.model['name'] = 'Classifier'
        self.model['in_c'] = 1  # input channel
        self.model['out_c'] = 3
        self.model['train_res4'] = True      # True: train res4, False: fixed all parameters
        self.model['resnet_layers'] = 18
        self.model['use_resnet'] = 0
        self.model['pre_train_sup'] = 0
        #self.model['sup_model_path']='../fibrosis_self_supervision/checkpoint/checkpoint_best.pth.tar'
        self.model['sup_model_path']=''

        # --- training params --- #
        self.train = dict()
        self.train['random_seed'] = 1
        #self.train['data_dir'] = './data/data_isbi'  # path to data
        self.train['data_dir'] = ''  # path to data
        self.train['save_dir'] = './experiments/{:s}/{:s}'.format(self.exp, self.exp_num)  # path to save results
        self.train['input_size'] = 224          # input size of the image
        self.train['epochs'] = 30         # number of slide-level training epochs
        self.train['batch_size'] = 4     # batch size of slide-level training
        self.train['checkpoint_freq'] = 60      # epoch to save checkpoints
        self.train['lr'] = 0.0001                # initial learning rate
        self.train['weight_decay'] = 0.01       # weight decay
        self.train['log_interval'] = 8         # iterations to print training results
        self.train['workers'] = 4               # number of workers to load images
        self.train['gpus'] = [0]              # select gpu devices
        # --- resume training --- #
        self.train['start_epoch'] = 0    # start epoch
        self.train['checkpoint'] = ''
        # self.train['checkpoint'] = '../experiments/{:s}/11_2/checkpoint_best_tile_repeat_0.pth.tar'.format(self.exp)

        # --- test parameters --- #
        self.test = dict()
        self.test['test_epoch'] = 'best'
        self.test['gpus'] = [0]
        self.test['batch_size'] = 8
        #self.test['img_dir'] = './data/data_isbi'
        self.test['img_dir'] = ''
        self.test['save_flag'] = False
        self.test['save_dir'] = './experiments/{:s}/{:s}/{:s}'.format(self.exp, self.exp_num, self.test['test_epoch'])
        self.test['model_path'] = './experiments/{:s}/{:s}/checkpoint_{:s}.pth.tar'.format(self.exp, self.exp_num, self.test['test_epoch'])

    def parse(self):
        """ Parse the options, replace the default value if there is a new input """
        parser = argparse.ArgumentParser(description='')
        if self.isTrain:
            parser.add_argument('--exp', type=str, default=self.exp, help='exp name')
            parser.add_argument('--exp-num', type=str, default=self.exp_num, help='exp num')
            parser.add_argument('--model', type=str, default=self.model['name'], help='model name')
            parser.add_argument('--use_resnet', type=int, default=self.model['use_resnet'], help='use resnet for feature extraciton')
            parser.add_argument('--pre_train_sup', type=int, default=self.model['pre_train_sup'], help='use resnet for feature extraciton')
            parser.add_argument('--num-class', type=int, default=self.model['out_c'], help='the number of classes')
            parser.add_argument('--random-seed', type=int, default=self.train['random_seed'], help='random seed for reproducibility')
            parser.add_argument('--epochs', type=int, default=self.train['epochs'], help='number of epochs to train')
            parser.add_argument('--batch-size', type=int, default=self.train['batch_size'],
                                help='input batch size for training')
            parser.add_argument('--lr', type=float, default=self.train['lr'], help='learning rate')
            parser.add_argument('--log-interval', type=int, default=self.train['log_interval'], help='how many batches to wait before logging training status')
            parser.add_argument('--gpus', type=int, nargs='+', default=self.train['gpus'], help='GPUs for training')
            parser.add_argument('--data-dir', type=str, default=self.train['data_dir'], help='directory of training data')
            parser.add_argument('--save-dir', type=str, default=self.train['save_dir'], help='directory to save training results')
            parser.add_argument('--sup_model_path', type=str, default=self.model['sup_model_path'], help='directory to self-supervised model checkpoint')
            args = parser.parse_args()

            self.exp = args.exp
            self.exp_num = args.exp_num
            self.model['name'] = args.model
            self.model['use_resnet'] = args.use_resnet
            self.model['pre_train_sup'] = args.pre_train_sup
            self.model['out_c'] = args.num_class
            self.train['random_seed'] = args.random_seed
            self.train['batch_size'] = args.batch_size
            self.train['epochs'] = args.epochs
            self.train['lr'] = args.lr
            self.train['log_interval'] = args.log_interval
            self.train['gpus'] = args.gpus
            self.train['data_dir'] = args.data_dir
            print('data_dir: ', self.train['data_dir'])
            self.train['save_dir'] = args.save_dir
            self.model['sup_model_path'] = args.sup_model_path
            if not os.path.exists(self.train['save_dir']):
                os.makedirs(self.train['save_dir'], exist_ok=True)

        else:
            parser.add_argument('--exp', type=str, default=self.exp, help='exp name')
            parser.add_argument('--exp-num', type=str, default=self.exp_num, help='exp num')
            parser.add_argument('--model', type=str, default=self.model['name'], help='model name')
            parser.add_argument('--use_resnet', type=int, default=self.model['use_resnet'], help='use resnet for feature extraciton')
            parser.add_argument('--pre_train_sup', type=int, default=self.model['pre_train_sup'], help='use resnet for feature extraciton')
            parser.add_argument('--num-class', type=int, default=self.model['out_c'], help='the number of classes')
            parser.add_argument('--gpus', type=int, nargs='+', default=self.test['gpus'], help='GPUs for test')
            parser.add_argument('--batch-size', type=int, default=self.test['batch_size'], help='input batch size for test')
            parser.add_argument('--save-flag', type=bool, default=self.test['save_flag'], help='flag to save the network outputs and predictions')
            parser.add_argument('--img-dir', type=str, default=self.test['img_dir'], help='directory of test images')
            parser.add_argument('--save-dir', type=str, default=self.test['save_dir'], help='directory to save test results')
            parser.add_argument('--model-path', type=str, default=self.test['model_path'], help='train model to be evaluated')
            args = parser.parse_args()
            self.exp = args.exp
            self.exp_num = args.exp_num
            self.model['name'] = args.model
            self.model['use_resnet'] = args.use_resnet
            self.model['pre_train_sup'] = args.pre_train_sup
            self.model['out_c'] = args.num_class
            self.test['gpus'] = args.gpus
            self.test['batch_size'] = args.batch_size
            self.test['save_flag'] = args.save_flag
            self.test['img_dir'] = args.img_dir
            print('img_dir: ', self.test['img_dir'])
            self.test['save_dir'] = args.save_dir
            self.test['model_path'] = args.model_path

            if not os.path.exists(self.test['save_dir']):
                os.makedirs(self.test['save_dir'], exist_ok=True)

    def print_options(self, logger=None):
        message = '\n'
        message += self._generate_message_from_options()
        if not logger:
            print(message)
        else:
            logger.info(message)

    def save_options(self):
        if self.isTrain:
            filename = '{:s}/train_options.txt'.format(self.train['save_dir'])
        else:
            filename = '{:s}/test_options.txt'.format(self.test['save_dir'])
        message = self._generate_message_from_options()
        file = open(filename, 'w')
        file.write(message)
        file.close()

    def _generate_message_from_options(self):
        message = ''
        message += '# {str:s} Options {str:s} #\n'.format(str='-' * 25)
        train_groups = ['model', 'train', 'transform']
        test_groups = ['model', 'test', 'transform']
        cur_group = train_groups if self.isTrain else test_groups

        for group, options in self.__dict__.items():
            if group not in train_groups + test_groups:
                message += '{:>15}: {:<35}\n'.format(group, str(options))
            elif group in cur_group:
                message += '\n{:s} {:s} {:s}\n'.format('*' * 15, group, '*' * 15)
                if group == 'transform':
                    for name, val in options.items():
                        # message += '{:s}:\n'.format(name)
                        val = str(val).replace('\n', ',\n{:17}'.format(''))
                        message += '{:>15}: {:<35}\n'.format(name, str(val))
                else:
                    for name, val in options.items():
                        message += '{:>15}: {:<35}\n'.format(name, str(val))
        message += '# {str:s} End {str:s} #\n'.format(str='-' * 26)
        return message


