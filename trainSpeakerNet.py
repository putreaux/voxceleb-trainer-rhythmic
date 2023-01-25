#!/usr/bin/python
#-*- coding: utf-8 -*-

import sys, time, os, argparse
import yaml
import numpy
import torch
import glob
import zipfile
import warnings
import datetime
from tuneThreshold import *
from SpeakerNet import *
from DatasetLoader import *
import torch.distributed as dist
import torch.multiprocessing as mp
from torch.utils.tensorboard import SummaryWriter
warnings.simplefilter("ignore")
import multiprocessing

## ===== ===== ===== ===== ===== ===== ===== =====
## Parse arguments
## ===== ===== ===== ===== ===== ===== ===== =====

parser = argparse.ArgumentParser(description = "SpeakerNet")

parser.add_argument('--config',         type=str,   default=None,   help='Config YAML file')

## Data loader
parser.add_argument('--max_frames',     type=int,   default=200,    help='Input length to the network for training')
parser.add_argument('--eval_frames',    type=int,   default=300,    help='Input length to the network for testing 0 uses the whole files')
parser.add_argument('--batch_size',     type=int,   default=32,    help='Batch size, number of speakers per batch')
parser.add_argument('--max_seg_per_spk', type=int,  default=50,    help='Maximum number of utterances per speaker per epoch')
parser.add_argument('--nDataLoaderThread', type=int, default=5,     help='Number of loader threads')
parser.add_argument('--augment',        type=bool,  default=False,  help='Augment input')
parser.add_argument('--seed',           type=int,   default=10,     help='Seed for the random number generator')

## Training details
parser.add_argument('--test_interval',  type=int,   default=10,     help='Test and save every [test_interval] epochs')
parser.add_argument('--max_epoch',      type=int,   default=100,    help='Maximum number of epochs')
parser.add_argument('--trainfunc',      type=str,   default="",     help='Loss function')

## Optimizer
parser.add_argument('--optimizer',      type=str,   default="adam", help='sgd or adam')
parser.add_argument('--scheduler',      type=str,   default="steplr", help='Learning rate scheduler')
parser.add_argument('--lr',             type=float, default=0.001,  help='Learning rate')
parser.add_argument("--lr_decay",       type=float, default=0.95,   help='Learning rate decay every [test_interval] epochs')
parser.add_argument('--weight_decay',   type=float, default=0.0001,      help='Weight decay in the optimizer')

## Loss functions
parser.add_argument("--hard_prob",      type=float, default=0.5,    help='Hard negative mining probability, otherwise random, only for some loss functions')
parser.add_argument("--hard_rank",      type=int,   default=10,     help='Hard negative mining rank in the batch, only for some loss functions')
parser.add_argument('--margin',         type=float, default=0.1,    help='Loss margin, only for some loss functions')
parser.add_argument('--scale',          type=float, default=30,     help='Loss scale, only for some loss functions')
parser.add_argument('--nPerSpeaker',    type=int,   default=1,      help='Number of utterances per speaker per batch, only for metric learning based losses')
parser.add_argument('--nClasses',       type=int,   default=863,     help='Number of speakers in the softmax layer, only for softmax-based losses')

## Evaluation parameters
parser.add_argument('--dcf_p_target',   type=float, default=0.05,   help='A priori probability of the specified target speaker')
parser.add_argument('--dcf_c_miss',     type=float, default=1,      help='Cost of a missed detection')
parser.add_argument('--dcf_c_fa',       type=float, default=1,      help='Cost of a spurious detection')
parser.add_argument('--verify',         type=bool,  default=False,  help='Evaluation is either verification or identification')

## Load and save
parser.add_argument('--initial_model',  type=str,   default="",     help='Initial model weights')
parser.add_argument('--save_path',      type=str,   default="exps/exp1", help='Path for model and logs')

## Training and test data
parser.add_argument('--train_list',     type=str,   default="data/train_list.txt",  help='Train list')
parser.add_argument('--test_list',      type=str,   default="data/test_list.txt",   help='Evaluation list')
parser.add_argument('--train_path',     type=str,   default="data/train", help='Absolute path to the train set')
parser.add_argument('--test_path',      type=str,   default="data/train", help='Absolute path to the test set')
parser.add_argument('--musan_path',     type=str,   default="data/musan_split", help='Absolute path to the test set')
parser.add_argument('--rir_path',       type=str,   default="data/RIRS_NOISES/simulated_rirs", help='Absolute path to the test set')

## Model definition
parser.add_argument('--n_mels',         type=int,   default=40,     help='Number of mel filterbanks')
parser.add_argument('--log_input',      type=bool,  default=False,  help='Log input features')
parser.add_argument('--model',          type=str,   default="",     help='Name of model definition')
parser.add_argument('--encoder_type',   type=str,   default="SAP",  help='Type of encoder')
parser.add_argument('--nOut',           type=int,   default=100,    help='Embedding size in the last FC layer')
parser.add_argument('--sinc_stride',    type=int,   default=10,    help='Stride size of the first analytic filterbank layer of RawNet3')

## For test only
parser.add_argument('--eval',           dest='eval', action='store_true', help='Eval only')

## Distributed and mixed precision training
parser.add_argument('--port',           type=str,   default="8888", help='Port for distributed training, input as text')
parser.add_argument('--distributed',    dest='distributed', action='store_true', help='Enable distributed training')
parser.add_argument('--mixedprec',      dest='mixedprec',   action='store_true', help='Enable mixed precision training')

args = parser.parse_args()

writer = SummaryWriter()

## Parse YAML
def find_option_type(key, parser):
    for opt in parser._get_optional_actions():
        if ('--' + key) in opt.option_strings:
           return opt.type
    raise ValueError

if args.config is not None:
    with open(args.config, "r") as f:
        yml_config = yaml.load(f, Loader=yaml.FullLoader)
    for k, v in yml_config.items():
        if k in args.__dict__:
            typ = find_option_type(k, parser)
            args.__dict__[k] = typ(v)
        else:
            sys.stderr.write("Ignored unknown parameter {} in yaml.\n".format(k))


## ===== ===== ===== ===== ===== ===== ===== =====
## Trainer script
## ===== ===== ===== ===== ===== ===== ===== =====

def main_worker(gpu, ngpus_per_node, args):

    args.gpu = gpu

    ## Load models
    s = SpeakerNet(**vars(args))

    if args.distributed:
        os.environ['MASTER_ADDR']='localhost'
        os.environ['MASTER_PORT']=args.port

        dist.init_process_group(backend='nccl', world_size=ngpus_per_node, rank=args.gpu)

        torch.cuda.set_device(args.gpu)
        s.cuda(args.gpu)

        s = torch.nn.parallel.DistributedDataParallel(s, device_ids=[args.gpu], find_unused_parameters=True)

        print('Loaded the model on GPU {:d}'.format(args.gpu))

    else:
        s = WrappedModel(s).cuda(args.gpu)

    it = 1
    eers = [100]

    if args.gpu == 0:
        ## Write args to scorefile
        scorefile   = open(args.result_save_path+"/scores.txt", "a+")

    #number of utterances per each of the 863 speakers, used for weighted random sampling
    class_counts = [52, 125, 134, 94, 97, 74, 93, 90, 120, 80, 58, 115, 66, 117, 107, 63, 66, 80, 106, 67, 157, 67, 54, 66, 116, 97,
     125, 138, 71, 60, 135, 54, 156, 130, 128, 58, 56, 168, 111, 97, 161, 99, 136, 123, 80, 97, 63, 90, 101, 57, 142,
     99, 100, 89, 100, 100, 123, 58, 110, 83, 79, 115, 68, 52, 100, 96, 127, 67, 77, 64, 78, 98, 128, 53, 157, 61, 141,
     95, 79, 67, 63, 89, 71, 100, 79, 126, 53, 142, 70, 72, 101, 62, 122, 57, 59, 130, 94, 98, 162, 53, 88, 56, 57, 95,
     90, 66, 75, 52, 69, 78, 79, 181, 176, 86, 51, 85, 73, 91, 78, 98, 140, 64, 70, 134, 117, 66, 98, 81, 72, 76, 54,
     76, 112, 145, 195, 60, 61, 85, 63, 67, 145, 109, 59, 100, 74, 84, 80, 93, 126, 110, 79, 79, 97, 119, 156, 61, 98,
     74, 178, 55, 118, 62, 88, 56, 62, 78, 74, 92, 155, 116, 92, 128, 108, 120, 70, 58, 60, 84, 84, 73, 101, 101, 82,
     51, 91, 53, 53, 64, 91, 82, 102, 84, 52, 57, 173, 131, 51, 157, 67, 90, 54, 93, 74, 112, 77, 60, 83, 76, 53, 55,
     63, 84, 54, 51, 134, 103, 96, 131, 113, 83, 123, 77, 67, 94, 106, 124, 172, 135, 132, 51, 59, 187, 163, 53, 138,
     60, 110, 95, 171, 81, 166, 178, 109, 57, 166, 76, 76, 97, 151, 68, 69, 77, 92, 110, 79, 92, 122, 123, 105, 58, 131,
     80, 61, 82, 107, 116, 92, 116, 134, 76, 91, 65, 189, 147, 108, 57, 64, 79, 117, 170, 66, 82, 111, 172, 114, 87, 75,
     193, 86, 138, 90, 96, 118, 57, 75, 82, 120, 56, 61, 52, 110, 100, 82, 56, 91, 131, 51, 100, 109, 157, 72, 52, 119,
     112, 67, 168, 152, 193, 88, 59, 59, 83, 75, 88, 58, 120, 128, 66, 77, 84, 68, 101, 122, 191, 54, 73, 87, 93, 115,
     87, 121, 68, 94, 127, 62, 80, 77, 86, 62, 77, 105, 100, 90, 161, 96, 97, 148, 56, 75, 106, 78, 64, 93, 56, 53, 82,
     107, 72, 70, 111, 52, 51, 82, 75, 152, 57, 119, 175, 51, 78, 71, 68, 136, 110, 107, 82, 82, 102, 80, 82, 55, 70,
     64, 105, 95, 74, 70, 66, 81, 86, 129, 74, 113, 89, 57, 71, 59, 102, 121, 141, 51, 51, 127, 141, 67, 119, 87, 74,
     88, 103, 63, 125, 53, 160, 73, 66, 112, 164, 78, 52, 56, 79, 122, 77, 73, 86, 61, 72, 54, 119, 69, 123, 60, 68,
     155, 81, 75, 65, 102, 90, 96, 109, 66, 60, 138, 90, 63, 95, 168, 63, 76, 71, 122, 51, 89, 66, 59, 70, 57, 67, 148,
     150, 185, 127, 155, 59, 51, 64, 60, 60, 185, 180, 65, 52, 120, 93, 124, 151, 75, 99, 53, 113, 90, 70, 72, 128, 117,
     69, 71, 134, 179, 52, 141, 101, 76, 132, 88, 112, 183, 68, 195, 189, 108, 148, 76, 51, 79, 89, 101, 130, 56, 186,
     79, 53, 59, 60, 109, 102, 79, 60, 78, 72, 115, 127, 191, 54, 65, 188, 133, 96, 68, 88, 86, 56, 173, 115, 187, 114,
     138, 117, 58, 51, 122, 56, 130, 58, 100, 83, 92, 101, 159, 86, 79, 81, 89, 72, 57, 75, 155, 103, 63, 81, 54, 160,
     111, 76, 91, 93, 104, 55, 117, 107, 157, 52, 122, 114, 57, 90, 53, 82, 140, 57, 124, 62, 60, 71, 68, 91, 108, 75,
     94, 191, 95, 132, 52, 86, 187, 59, 72, 78, 138, 103, 86, 69, 156, 137, 113, 100, 168, 76, 61, 138, 100, 82, 69, 52,
     182, 199, 186, 119, 105, 57, 107, 92, 80, 73, 51, 71, 118, 190, 52, 142, 152, 64, 55, 101, 55, 130, 62, 153, 194,
     176, 126, 64, 122, 81, 58, 64, 59, 95, 78, 69, 118, 54, 53, 113, 102, 114, 84, 52, 72, 92, 74, 56, 57, 72, 60, 184,
     133, 58, 74, 81, 149, 144, 196, 85, 75, 145, 161, 66, 89, 113, 64, 60, 123, 64, 128, 92, 55, 83, 85, 107, 80, 175,
     161, 92, 99, 110, 56, 65, 70, 55, 60, 163, 97, 79, 70, 173, 92, 101, 123, 79, 87, 87, 56, 109, 53, 59, 149, 67, 55,
     61, 71, 163, 90, 79, 103, 178, 98, 76, 161, 91, 95, 71, 77, 138, 184, 128, 168, 77, 84, 124, 69, 96, 175, 58, 132,
     93, 97, 102, 71, 51, 106, 75, 52, 66, 84, 135, 130, 54, 61, 67, 99, 146, 189, 121, 59, 75, 127, 69, 98, 102, 63,
     114, 53, 52, 58, 71, 58, 114, 60, 120, 104, 106, 178, 110, 120, 87, 65, 80, 69, 146, 98, 83, 161, 62, 108, 68, 187,
     165, 53, 79, 151, 104, 128, 196, 162, 67, 152, 107, 56, 62, 69, 56, 86, 70, 86, 72, 58, 76, 89, 94, 89, 53, 109,
     124, 87, 68, 89, 78, 63, 63, 93, 63, 119, 63, 90, 185, 102, 94, 178, 94, 125, 114, 62, 87, 116, 130, 69, 100, 160,
     172, 59]

    ## Initialise trainer and data loader
    train_dataset = train_feature_loader(**vars(args))
    val_lines = train_dataset.get_val_lines()
    validation_dataset = val_feature_loader(**vars(args), lines=val_lines)
    train_labels = train_dataset.get_labels()
    val_labels = validation_dataset.get_labels()
    weight = 1 / torch.Tensor(class_counts)
    label_weights = [weight[i] for i in train_labels]

    rand_sampler = torch.utils.data.WeightedRandomSampler(label_weights, len(label_weights), replacement=True)

    train_sampler = train_dataset_sampler(train_dataset, **vars(args))

    train_loader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=64,
        shuffle=True,
        num_workers=multiprocessing.cpu_count()-1,
        sampler=None,
        pin_memory=False,
        worker_init_fn=worker_init_fn,
        drop_last=True,
    )
    val_loader = torch.utils.data.DataLoader(
        validation_dataset,
        batch_size=32,
        shuffle=False,
        num_workers=multiprocessing.cpu_count() - 1,
        pin_memory=True,
        drop_last=True,
    )
    trainer     = ModelTrainer(s, **vars(args))


    ## Load model weights
    modelfiles = glob.glob('%s/model0*.model'%args.model_save_path)
    modelfiles.sort()

    if(args.initial_model != ""):
        trainer.loadParameters(args.initial_model)
        print("Model {} loaded!".format(args.initial_model))
    elif len(modelfiles) >= 1:
        trainer.loadParameters(modelfiles[-1])
        print("Model {} loaded from previous state!".format(modelfiles[-1]))
        it = int(os.path.splitext(os.path.basename(modelfiles[-1]))[0][5:]) + 1

    for ii in range(1,it):
        trainer.__scheduler__.step()

    ## Evaluation code - must run on single GPU
    if args.eval == True:

        pytorch_total_params = sum(p.numel() for p in s.module.__S__.parameters())

        ## Initialize test loader
        test_id_dataset = test_dataset_loader_for_identification(**vars(args))



        print('Total parameters: ',pytorch_total_params)
        print('Test list',args.test_list)

        sc, lab, _ = trainer.evaluateFromList(**vars(args))

        if args.gpu == 0 and args.verify == True:

            result = tuneThresholdfromScore(sc, lab, [1, 0.1])

            fnrs, fprs, thresholds = ComputeErrorRates(sc, lab)
            mindcf, threshold = ComputeMinDcf(fnrs, fprs, thresholds, args.dcf_p_target, args.dcf_c_miss, args.dcf_c_fa)

            print('\n',time.strftime("%Y-%m-%d %H:%M:%S"), "VEER {:2.4f}".format(result[1]), "MinDCF {:2.5f}".format(mindcf))
        
        if args.gpu == 0 and args.verify == False:
            loss, acc = trainer.evaluateForIdentification(args.test_list, **kwargs)
        return

    ## Save training code and params
    if args.gpu == 0:
        pyfiles = glob.glob('./*.py')
        strtime = datetime.datetime.now().strftime("%Y%m%d%H%M%S")

        zipf = zipfile.ZipFile(args.result_save_path+ '/run%s.zip'%strtime, 'w', zipfile.ZIP_DEFLATED)
        for file in pyfiles:
            zipf.write(file)
        zipf.close()

        with open(args.result_save_path + '/run%s.cmd'%strtime, 'w') as f:
            f.write('%s'%args)

    ## Core training script
    for it in range(it,args.max_epoch+1):

        train_sampler.set_epoch(it)
        clr = [x['lr'] for x in trainer.__optimizer__.param_groups]

        loss, traineer = trainer.train_network(train_loader, verbose=(args.gpu == 0))
        writer.add_scalar("Loss/train", loss, it)
        writer.add_scalar("Acc/train", traineer, it)

        if args.gpu == 0:
            print('\n',time.strftime("%Y-%m-%d %H:%M:%S"), "Epoch {:d}, TEER/TAcc {:2.2f}, TLOSS {:f}, LR {:f}".format(it, traineer, loss, max(clr)))
            scorefile.write("Epoch {:d}, TEER/TAcc {:2.2f}, TLOSS {:f}, LR {:f} \n".format(it, traineer, loss, max(clr)))

        if it % args.test_interval == 5 and args.verify==True:

            #sc, lab, _ = trainer.evaluateFromList(**vars(args))

            if args.gpu == 0:

                result = tuneThresholdfromScore(sc, lab, [1, 0.1])

                fnrs, fprs, thresholds = ComputeErrorRates(sc, lab)
                mindcf, threshold = ComputeMinDcf(fnrs, fprs, thresholds, args.dcf_p_target, args.dcf_c_miss, args.dcf_c_fa)

                eers.append(result[1])

                print('\n',time.strftime("%Y-%m-%d %H:%M:%S"), "Epoch {:d}, VEER {:2.4f}, MinDCF {:2.5f}".format(it, result[1], mindcf))
                scorefile.write("Epoch {:d}, VEER {:2.4f}, MinDCF {:2.5f}\n".format(it, result[1], mindcf))

                trainer.saveParameters(args.model_save_path+"/model%09d.model"%it)

                with open(args.model_save_path+"/model%09d.eer"%it, 'w') as eerfile:
                    eerfile.write('{:2.4f}'.format(result[1]))

                scorefile.flush()
        val_loss, val_acc = trainer.validateForIdentification(val_labels, validation_dataset, val_loader)
        writer.add_scalar("Loss/validation", val_loss, it)
        writer.add_scalar("Acc/validation", val_acc, it)
        if it % 5 == 0 and args.verify==False:
            loss, acc = trainer.evaluateForIdentification(**vars(args))
            writer.add_scalar("Loss/test", loss, it)
            writer.add_scalar("Acc/test", acc, it)
            if args.gpu == 0:
               print('\n',time.strftime("%Y-%m-%d %H:%M:%S"), "Epoch {:d}, Test Accuracy {:2.2f}, Test Loss {:f}".format(it, acc, loss))
               scorefile.write("Epoch {:d}, Test Acc {:2.2f}, Test Loss {:f}\n".format(it, acc, loss))
            trainer.saveParameters(args.model_save_path+"/model%09d.model"%it)

    writer.flush()
    writer.close()
    if args.gpu == 0:
        scorefile.close()


## ===== ===== ===== ===== ===== ===== ===== =====
## Main function
## ===== ===== ===== ===== ===== ===== ===== =====


def main():
    args.model_save_path     = args.save_path+"/model"
    args.result_save_path    = args.save_path+"/result"
    args.feat_save_path      = ""

    os.makedirs(args.model_save_path, exist_ok=True)
    os.makedirs(args.result_save_path, exist_ok=True)

    n_gpus = torch.cuda.device_count()

    print('Python Version:', sys.version)
    print('PyTorch Version:', torch.__version__)
    print('Number of GPUs:', torch.cuda.device_count())
    print('Save path:',args.save_path)

    if args.distributed:
        mp.spawn(main_worker, nprocs=n_gpus, args=(n_gpus, args))
    else:
        main_worker(0, None, args)


if __name__ == '__main__':
    main()
