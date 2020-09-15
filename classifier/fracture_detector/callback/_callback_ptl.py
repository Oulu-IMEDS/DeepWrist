from pytorch_lightning import Callback


class ReleaseAfterCallback(Callback):
    def __init__(self, release_after):
        self.__epoch = 0
        self.__release_after = release_after

    def on_epoch_start(self, trainer, pl_module):
        self.__epoch += 1

        if self.__epoch == self.__release_after + 1:
            print('\n\nReleasing the full model')
            opt, sched = trainer.model.configure_optimizers(classifier_only=False)
            trainer.optimizers, trainer.lr_schedulers[0]['scheduler'] = opt, sched[0]
            # trainer.optimizers[0].add_param_group({'params': trainer.model.encoder.parameters()})

