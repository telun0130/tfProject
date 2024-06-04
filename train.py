import os
from matplotlib import pyplot as plt
from model import ModelCreator
from data import DScreate

os.environ["CUDA_VISIBLE_DEVICES"] = "0"

class trainer():
    def __init__(self, TrainData, TrainLabel, ValidData = None, ValidLabel = None, plot = False):
        self.TrainPath = [TrainData, TrainLabel]
        self.ValidPath = [ValidData, ValidLabel]
        self.plot = plot
    def train(self, modelname, channel_in, channel_target, valid = False, epoch = 5, batch=1):
        trainset = DScreate(self.TrainPath[0], self.TrainPath[1])
        modelcaller = ModelCreator()
        Model = modelcaller.BuildModel(ch_in=channel_in, num_class=channel_target, option=modelname)
        if Model[0] == True:
            if self.ValidPath[0] is None or self.ValidPath[1] is None:
                history = Model[1].fit(trainset, verbose=1, epochs=epoch, batch_size=batch)
            else:
                validset = DScreate(self.ValidPath[0], self.ValidPath[1])
                history = Model[1].fit(trainset, verbose=1, epochs=epoch, batch_size=batch, validation_data=validset)
        else:
            print(Model[1])
            return 0
        if valid != False:
            if self.plot:
                train_loss = history.history['loss']
                valid_loss = history.history['val_loss']
                train_iou = history.history['io_u']
                valid_iou = history.history['val_io_u']
                train_acc = history.history['accuracy']
                valid_acc = history.history['val_accuracy']
                epoch_idx = list(range(1, epoch+1))
                plt.subplot(2, 2, 1)
                plt.xlabel('Epoch')
                plt.ylabel('Loss')
                plt.plot(epoch_idx, train_loss)
                plt.plot(epoch_idx, valid_loss)
                plt.legend(['train', 'valid'])

                plt.subplot(2, 2, 2)
                plt.xlabel('Epoch')
                plt.ylabel('IOU')
                plt.plot(epoch_idx, train_iou)
                plt.plot(epoch_idx, valid_iou)
                plt.legend(['train', 'valid'])

                plt.subplot(2, 2, 3)
                plt.xlabel('Epoch')
                plt.ylabel('Acc')
                plt.plot(epoch_idx, train_acc)
                plt.plot(epoch_idx, valid_acc)
                plt.legend(['train', 'valid'])
                plt.savefig('test.png')
                plt.show()
        else:
            if self.plot:
                train_loss = history.history['loss']
                train_iou = history.history['io_u']
                train_acc = history.history['accuracy']
                epoch_idx = list(range(1, epoch+1))
                plt.subplot(2, 2, 1)
                plt.xlabel('Epoch')
                plt.ylabel('Loss')
                plt.plot(epoch_idx, train_loss)

                plt.subplot(2, 2, 2)
                plt.xlabel('Epoch')
                plt.ylabel('IOU')
                plt.plot(epoch_idx, train_iou)

                plt.subplot(2, 2, 3)
                plt.xlabel('Epoch')
                plt.ylabel('Acc')
                plt.plot(epoch_idx, train_acc)
                plt.savefig('test.png')
                plt.show()

    def trainwithsave(self, savedest, modelname, channel_in, channel_target, valid = False, epoch=5, batch=1):
        trainset = DScreate(self.TrainPath[0], self.TrainPath[1])
        modelcaller = ModelCreator()
        Model = modelcaller.BuildModel(ch_in=channel_in, num_class=channel_target, option=modelname)
        if Model[0] == True:
            if self.ValidPath[0] is None or self.ValidPath[1] is None:
                history = Model[1].fit(trainset, verbose=1, epochs=epoch, batch_size=batch)
            else:
                validset = DScreate(self.ValidPath[0], self.ValidPath[1])
                history = Model[1].fit(trainset, verbose=1, epochs=epoch, batch_size=batch, validation_data=validset)
        else:
            print(Model[1])
            return 0
        if valid != False:
            if self.plot:
                train_loss = history.history['loss']
                valid_loss = history.history['val_loss']
                train_iou = history.history['io_u']
                valid_iou = history.history['val_io_u']
                train_acc = history.history['accuracy']
                valid_acc = history.history['val_accuracy']
                epoch_idx = list(range(1, epoch+1))
                plt.subplot(2, 2, 1)
                plt.xlabel('Epoch')
                plt.ylabel('Loss')
                plt.plot(epoch_idx, train_loss)
                plt.plot(epoch_idx, valid_loss)
                plt.legend(['train', 'valid'])

                plt.subplot(2, 2, 2)
                plt.xlabel('Epoch')
                plt.ylabel('IOU')
                plt.plot(epoch_idx, train_iou)
                plt.plot(epoch_idx, valid_iou)
                plt.legend(['train', 'valid'])

                plt.subplot(2, 2, 3)
                plt.xlabel('Epoch')
                plt.ylabel('Acc')
                plt.plot(epoch_idx, train_acc)
                plt.plot(epoch_idx, valid_acc)
                plt.legend(['train', 'valid'])
                plt.savefig('test.png')
                plt.show()
        else:
            if self.plot:
                train_loss = history.history['loss']
                train_iou = history.history['io_u']
                train_acc = history.history['accuracy']
                epoch_idx = list(range(1, epoch+1))
                plt.subplot(2, 2, 1)
                plt.xlabel('Epoch')
                plt.ylabel('Loss')
                plt.plot(epoch_idx, train_loss)

                plt.subplot(2, 2, 2)
                plt.xlabel('Epoch')
                plt.ylabel('IOU')
                plt.plot(epoch_idx, train_iou)

                plt.subplot(2, 2, 3)
                plt.xlabel('Epoch')
                plt.ylabel('Acc')
                plt.plot(epoch_idx, train_acc)
                plt.savefig('test.png')
                plt.show()
        Model[1].save(os.path.join(savedest, f"{modelname}.keras"))

if __name__ == '__main__':
    Trainer = trainer(TrainData='train_1000/img', TrainLabel='train_1000/mask',plot = True)
    Trainer.train(modelname='R2U-Net', channel_in=1, channel_target=2, epoch=3)