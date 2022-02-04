import logging

from FedML.fedml_core.trainer.model_trainer import ModelTrainer


class FedSCAFFOLDTransformerTrainer(ModelTrainer):

    def __init__(self, trainer, model):
        super().__init__(model)
        self.model_trainer = trainer
        self.model = model

    def get_model_params(self):
        return self.model.cpu().state_dict()

    def set_model_params(self, model_parameters):
        self.model.load_state_dict(model_parameters)

    def train(self, train_data, device, args, global_control_variate, local_control_variate):
        logging.info("Client(%d)" % self.id + ":| Local Train Data Size = %d" % (len(train_data)))
        self.model_trainer.train_dl = train_data
        _, _, local_control_variate = self.model_trainer.train_model(device=device,
                                                                     global_control_variate=global_control_variate,
                                                                     local_control_variate=local_control_variate)
        return local_control_variate

    def test(self, test_data, device, args=None):
        pass

    def test_on_the_server(self, train_data_local_dict, test_data_local_dict, device, args=None):
        self.model_trainer.eval_model(device=device)
        return True
