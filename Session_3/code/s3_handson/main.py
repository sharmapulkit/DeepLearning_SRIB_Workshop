#!/usr/bin/env python

import os
import argparse
from collections import namedtuple

from keras import optimizers
from keras.callbacks import TensorBoard, ModelCheckpoint, LearningRateScheduler

from dataloader import *
from model import *
from convert_to_png import *

parser = argparse.ArgumentParser()

parser.add_argument('--dataset_name', type=str)
parser.add_argument('--mode', type=str)
parser.add_argument('--model_name', type=str,
                    choices=['simple', 'deep_supervision'])
parser.add_argument('--datapath', type=str)
parser.add_argument('--train_list', type=str)
parser.add_argument('--val_list', type=str)
parser.add_argument('--test_list', type=str)
parser.add_argument('--crop_height', type=int, default=256)
parser.add_argument('--crop_width', type=int, default=512)
parser.add_argument('--batch_size', type=int, default=8)
parser.add_argument('--num_epochs', type=int, default=50)
parser.add_argument('--optimizer', type=str, default='adam',
                    choices=['adam', 'rms'])
parser.add_argument('--checkpoint', type=str, default='')
parser.add_argument('--start_epoch', type=int, default=0)
parser.add_argument('--learning_rate', type=float, default=2e-4)
parser.add_argument('--ngpu', type=int, default=1)
parser.add_argument('--num_threads', type=int, default=2)
parser.add_argument('--output_directory', type=str)

args = parser.parse_args()

""" learning rate schedular to change the learning rate during training """
def schedule(epoch):
    if epoch < 15:
        return np.float32(args.learning_rate)
    elif epoch < 25:
        return np.float32(args.learning_rate/2)
    elif epoch < 35:
        return np.float32(args.learning_rate/4)
    else:
        return np.float32(args.learning_rate/8)

def train(params):
    # train and validation data generators
    train_gen = Dataloader(params, args.train_list)
    validation_gen = Dataloader(params, args.val_list)
    
    if args.model_name in ['simple', 'deep_supervision']:
        dispmodel = DispModel(params)
        model = dispmodel.model
        l1_loss = dispmodel.l1_loss
    else:
        SystemExit("Wrong model!")
    model.summary()
    
    """ use for training the model on multiple gpus """
    # model = keras.utils.multi_gpu_model(model, gpus=2)

    """ schedular and optimizer """
    scheduler = LearningRateScheduler(schedule)
    if args.optimizer == 'adam':
        optimizer = optimizers.Adam(lr=args.learning_rate)
    elif args.optimizer == 'rms':
        optimizer = optimizers.RMSprop(lr=args.learning_rate)
    else:
        SystemExit('Optimizer not supported!')

    model.compile(loss=loss, optimizer=optimizer)

    # load an already saved model
    if args.checkpoint != '':
        model.load_weights(filepath=args.checkpoint)

        
    """ setting up directorues for checkpoints and log """
    if not (os.path.isdir('./checkpoint')):
        os.mkdir('./checkpoint')
    
    save_path = './checkpoint/' + str(args.model_name)
    if not (os.path.isdir(save_path)):
        os.mkdir(save_path)
        
    log_path = './checkpoint/' + str(args.model_name) + '/logs/'
    if not (os.path.isdir(log_path)):
        os.mkdir(log_path)

    """ checkpoint and tensorboard callbacks """
    checkpoint = ModelCheckpoint(filepath="./checkpoint/" + str(args.model_name) + "/weights.{epoch:02d}-{val_loss:.2f}.hdf5", period=1,
                                 save_weights_only=True)
    tensorboard = TensorBoard(log_dir=log_path, write_images=True)

    """ train the model! """
    model.fit_generator(train_gen, steps_per_epoch=train_gen.__len__(), epochs=params.num_epochs, verbose=1, max_queue_size=500,
                        use_multiprocessing=True, callbacks=[scheduler, checkpoint, tensorboard],
                        validation_data=validation_gen, validation_steps=validation_gen.__len__(), workers=params.num_threads, initial_epoch=args.start_epoch)

def test(params):
    test_gen = Dataloader(params, args.test_list)
    
    if args.model_name in ['simple', 'deep_supervision']:
        dispmodel = DispModel(params)
        model = dispmodel.model
        l1_loss = dispmodel.l1_loss
    else:
        SystemExit("Wrong model!")
    model.summary()
    
    """ optimizer """
    scheduler = LearningRateScheduler(schedule)
    if args.optimizer == 'adam':
        optimizer = optimizers.Adam(lr=args.learning_rate)
    elif args.optimizer == 'rms':
        optimizer = optimizers.RMSprop(lr=args.learning_rate)
    else:
        SystemExit('Optimizer not supported!')

    model.compile(loss=loss, optimizer=optimizer , metrics=["accuracy"])
    
    if args.checkpoint = '':
        SystemExit('Checkpoint not provided!')
    else:
        model.load_weights(filepath=args.checkpoint)
        
    x = model.predict_generator(test_gen,steps = test_gen.__len__(),max_queue_size=500,use_multiprocessing=True)
    #x = model.evaluate_generator(test_gen,steps = test_gen.__len__(),max_queue_size=10,use_multiprocessing=True,verbose=1)
    j = 0
    f = args.output_directory
    for i in x:
        print (j)
        writePFM(f + "/"+str(j)+".pfm",i)
        ConvertMiddlebury2014PfmToKitti2015Png(f + "/"+str(j)+".pfm", f + "/"+str(j)+".png")
        j = j + 1
        
def main():
    # some important parameters needed by other modules
    Params = namedtuple('Parameters',
                        'mode,'
                        'model_name,'
                        'dataset_name,'
                        'datapath,'
                        'crop_height,'
                        'crop_width,'
                        'batch_size,'
                        'ngpu,'
                        'num_threads,'
                        'num_epochs')

    params = Params(mode=args.mode,
                    model_name=args.model_name,
                    dataset_name=args.dataset_name,
                    datapath=args.datapath,
                    crop_height=args.crop_height,
                    crop_width=args.crop_width,
                    batch_size=args.batch_size,
                    ngpu=args.ngpu,
                    num_threads=args.num_threads,
                    num_epochs=args.num_epochs)

    if args.mode == 'train':
        train(params)
    elif args.mode == 'test':
        test(params)


if __name__ == '__main__':
    main()