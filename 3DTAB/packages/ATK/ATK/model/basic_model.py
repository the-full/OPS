from typing import Any

import torch
import pytorch_lightning as pl
import sklearn.metrics as metrics

from ATK.utils.common import return_first_item


class BasicModel(pl.LightningModule):
    def __init__(self):
        super().__init__()
        self.loss_function = torch.nn.CrossEntropyLoss()
        self.validation_step_outputs = []

    def training_step(self, batch, batch_idx): # type: ignore
        labels = batch['category'].view(-1)

        logits = self.__predict__(batch)
        loss = self.loss_function(logits, labels)
        self.log('loss', loss, on_step=True, on_epoch=True, prog_bar=True)
        return loss

    def validation_step(self, batch, batch_idx): # type: ignore
        labels = batch['category'].view(-1)

        logits = self.__predict__(batch)
        preds  = logits.argmax(dim=-1)
        loss = self.loss_function(logits, labels)

        self.log('val_loss', loss, on_step=False, on_epoch=True, prog_bar=False)
        self.validation_step_outputs.append((preds, labels))
        return loss

    def test_step(self, batch, batch_idx): # type: ignore
        # Here we just reuse the validation_step for testing
        return self.validation_step(batch, batch_idx)

    def on_validation_epoch_end(self):
        all_outputs = self.validation_step_outputs
        if all_outputs == []:
            pass
        else:
            all_preds = []
            all_labels = []
            for preds, labels in all_outputs:
                all_preds.extend(preds.tolist())
                all_labels.extend(labels.tolist())
            # NOTE: in sklearn y_true first.
            acc  = metrics.accuracy_score(all_labels, all_preds)
            bacc = metrics.balanced_accuracy_score(all_labels, all_preds)
            self.log('val_acc',  acc * 100,  prog_bar=True)
            self.log('val_bacc', bacc * 100, prog_bar=True)
            self.validation_step_outputs.clear()

    def on_test_epoch_end(self):
        self.on_validation_end()

    def configure_optimizers(self): # type: ignore
        optimizer = torch.optim.Adam(self.parameters(), lr=1e-3, weight_decay=2e-6)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, 
            T_max=self.trainer.max_epochs,  # type: ignore
            eta_min=1e-6
        )
        return {'optimizer': optimizer, 'lr_scheduler': scheduler}

    def forward(self, data_dict) -> Any: # type: ignore
        pass

    def __predict__(self, data_dict):
        out = self(data_dict)
        return return_first_item(out)
