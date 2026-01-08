import torch
from train.model import UNet3D
from train.metrics import dice_score, iou_score, precision_score, recall_score
from train.loss import MaskLoss, TverskyLoss
import logging
from tqdm import tqdm
import os


class EarlyStopping:
    def __init__(self, patience=10, min_delta=0.001):
        self.patience = patience
        self.min_delta = min_delta
        self.best_loss = 1000
        self.counter = 0

    def check_stop(self, val_loss):
        if val_loss < self.best_loss - self.min_delta:
            self.best_loss = val_loss
            self.counter = 0
        else:
            self.counter += 1
        return self.counter >= self.patience


class TrainEpoch:
    def __init__(
        self,
        output_path,
        train_dataset,
        val_dataset,
        lr: float,
        model,
        device,
        epoch_num: int,
        report_every: int = 10,
        scheduler_step_size=10,
        scheduler_gamma=0.2,
        weight_decay=0.0005,
    ):
        self.train_dataset = train_dataset
        self.val_dataset = val_dataset
        self.output_path = output_path
        self.lr = lr
        self.device = device
        self.epoch_num = epoch_num
        self.weight_decay = weight_decay
        self.report = report_every
        # Check if the training already ran
        files = [file for file in sorted(os.listdir(output_path)) if ".pt" in file]
        self.model = model.to(self.device)
        # Model optimizer
        self.optimizer = torch.optim.SGD(
            self.model.parameters(),
            lr=self.lr,
            momentum=0.9,
            weight_decay=weight_decay,
        )

        # Scheduler
        #self.scheduler = torch.optim.lr_scheduler.StepLR(
        #    self.optimizer, step_size=scheduler_step_size, gamma=scheduler_gamma
        #)
        
        #self.optimizer = torch.optim.RMSprop(model.parameters(), lr=self.lr, weight_decay=self.weight_decay, momentum=0.9)
        #self.optimizer = torch.optim.AdamW(self.model.parameters(),lr=self.lr, weight_decay=self.weight_decay)
        #self.scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(self.optimizer, mode='min', factor=0.2, patience=5)
        self.scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(self.optimizer, T_0=10, T_mult=2, eta_min=1e-5)
        if len(files) > 0:
            recent_ckpt = files[-1]
            self.resume_training(os.path.join(output_path, recent_ckpt))
        else:
            self.create_dict()

    def train_epoch(self):
        early_stopping = EarlyStopping(patience=10)
        # If the model has not run yet, epoch starts from 0
        if len(self.epoch_train_loss) == 0:
            self.start_epoch = 0
        # Loop through the epochs
        for epoch in range(self.start_epoch, self.epoch_num):
            train_avg_loss = 0
            train_avg_iou = 0
            train_avg_dice = 0
            train_avg_pre = 0
            train_avg_recall = 0

            val_avg_loss = 0
            val_avg_iou = 0
            val_avg_dice = 0
            val_avg_pre = 0
            val_avg_recall = 0
            
            train_len = 0
            val_len = 0

            for idx, batch in tqdm(enumerate(self.train_dataset)):
                train_loss, train_iou, train_dice, train_pre, train_recall = self._run_train(batch)
                train_len += len(batch[1])
                train_avg_loss += train_loss.item() * len(batch[1])
                train_avg_iou += train_iou.item()
                train_avg_dice += train_dice.item()
                train_avg_pre += train_pre.item()
                train_avg_recall += train_recall.item()

            train_avg_loss = train_avg_loss / train_len
            train_avg_dice = train_avg_dice / train_len
            train_avg_iou = train_avg_iou / train_len
            train_avg_pre = train_avg_pre / train_len
            train_avg_recall = train_avg_recall / train_len

            self.epoch_train_dice_score.append(train_avg_dice)
            self.epoch_train_iou_score.append(train_avg_iou)
            self.epoch_train_loss.append(train_avg_loss)
            self.epoch_train_precision_score.append(train_avg_pre)
            self.epoch_train_recall_score.append(train_avg_recall)

            self.scheduler.step()

            for idx, batch in tqdm(enumerate(self.val_dataset)):
                val_loss, val_iou, val_dice, val_pre, val_recall = self._run_val(batch)
                #print(idx, val_pre.item(), len(batch[1]))
                val_len += len(batch[1])
                val_avg_iou += val_iou.item()
                val_avg_dice += val_dice.item()
                val_avg_pre += val_pre.item()
                val_avg_recall += val_recall.item()
                val_avg_loss += val_loss.item() * len(batch[1])
            val_avg_loss = val_avg_loss / val_len
            val_avg_dice = val_avg_dice / val_len
            val_avg_iou = val_avg_iou / val_len
            val_avg_pre = val_avg_pre / val_len
            val_avg_recall = val_avg_recall / val_len
            #self.scheduler.step(val_avg_loss)
            self.epoch_val_dice_score.append(val_avg_dice)
            self.epoch_val_iou_score.append(val_avg_iou)
            self.epoch_val_loss.append(val_avg_loss)
            self.epoch_val_precision_score.append(val_avg_pre)
            self.epoch_val_recall_score.append(val_avg_recall)

            if epoch % self.report == 0:
                logging.info(f"Epoch:{epoch}")
                logging.info(f"Train loss: {train_avg_loss}")
                logging.info(f"Train dice score: {train_avg_dice}")
                logging.info(f"Train iou score: {train_avg_iou}")
                logging.info(f"Train precision score: {train_avg_pre}")
                logging.info(f"Train recall score: {train_avg_recall}")

                logging.info(f"Validation loss: {val_avg_loss}")
                logging.info(f"Validation dice score: {val_avg_dice}")
                logging.info(f"Validation iou score: {val_avg_iou}")
                logging.info(f"Validation precision score: {val_avg_pre}")
                logging.info(f"Validation recall score: {val_avg_recall}")
                #self.save_epoch(epoch)
            check = early_stopping.check_stop(val_avg_loss)
            if early_stopping.counter == 0:
                self.save_epoch(epoch)
            if check:
                logging.info(f"Stopping early at epoch {epoch}")
                break
        
    def _run_train(self, batch):
        self.model.train()
        imgs, labels = batch
        imgs = imgs.to(torch.float16).to(self.device)
        labels = labels.to(self.device)
        with torch.amp.autocast("cuda", enabled=True):
            self.optimizer.zero_grad()
            logits = self.model(imgs.unsqueeze(1))
            logits = logits.squeeze(1)
            probs = torch.sigmoid(logits)
            pred = torch.where(probs > 0.5, 1, 0).int()
            mask_loss = TverskyLoss()
            loss = mask_loss(logits,labels)
            loss.backward()
            self.optimizer.step()

        iou = iou_score(labels, pred)
        dice = dice_score(labels, pred)
        precision = precision_score(labels, pred)
        recall = recall_score(labels, pred)

        imgs.cpu()
        labels.cpu()
        logits.cpu()
        return loss, iou, dice, precision, recall

    @torch.no_grad()
    def _run_val(self, batch):
        self.model.eval()
        
        imgs, labels = batch
        imgs = imgs.to(self.device)
        labels = labels.to(self.device)
        logits = self.model(imgs.unsqueeze(1))
        logits = logits.squeeze(1)
        probs = torch.sigmoid(logits)
        pred = torch.where(probs > 0.5, 1, 0).int()
        mask_loss = TverskyLoss()
        loss = mask_loss(logits,labels)

        iou = iou_score(labels, pred)
        dice = dice_score(labels, pred)
        precision = precision_score(labels, pred)
        recall = recall_score(labels, pred)

        imgs.cpu()
        labels.cpu()
        logits.cpu()
        return loss, iou, dice, precision, recall

    def save_epoch(self, epoch):
        checkpoint = {
            "epoch": self.epoch_num,
            "model_state_dict": self.model.state_dict(),
            "optimizer_state_dict": self.optimizer.state_dict(),
            "scheduler_state_dict": self.scheduler.state_dict(),  # optional
            "train_loss": self.epoch_train_loss,  # last training loss (optional),
            "train_iou": self.epoch_train_iou_score,
            "train_dice": self.epoch_train_dice_score,
            "train_pre": self.epoch_train_precision_score,
            "train_recall": self.epoch_train_recall_score,
            "val_loss": self.epoch_val_loss,  # last training loss (optional),
            "val_iou": self.epoch_val_iou_score,
            "val_dice": self.epoch_val_dice_score,
            "val_pre": self.epoch_val_precision_score,
            "val_recall": self.epoch_val_recall_score,
        }
        torch.save(checkpoint, f"{self.output_path}/checkpoint_best.pth")

    def resume_training(self, checkpoint_path):
        checkpoint = torch.load(checkpoint_path, map_location=self.device)

        self.model.load_state_dict(checkpoint["model_state_dict"])
        self.optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        self.scheduler.load_state_dict(checkpoint["scheduler_state_dict"])

        self.epoch_train_loss = checkpoint["train_loss"]
        self.epoch_train_iou_score = checkpoint["train_iou"]
        self.epoch_train_dice_score = checkpoint["train_dice"]
        self.epoch_train_precision_score = checkpoint["train_pre"]
        self.epoch_train_recall_score = checkpoint["train_recall"]
        self.epoch_val_loss = checkpoint["val_loss"]
        self.epoch_val_iou_score = checkpoint["val_iou"]
        self.epoch_val_dice_score = checkpoint["val_dice"]
        self.epoch_val_precision_score = checkpoint["val_pre"]
        self.epoch_val_recall_score = checkpoint["val_recall"]

        self.start_epoch = len(self.epoch_train_loss) + 1
        logging(f"Resuming training at epoch {self.start_epoch}")

    def create_dict(self):
        self.epoch_train_loss = []
        self.epoch_train_dice_score = []
        self.epoch_train_iou_score = []
        self.epoch_train_precision_score = []
        self.epoch_train_recall_score = []

        self.epoch_val_loss = []
        self.epoch_val_dice_score = []
        self.epoch_val_iou_score = []
        self.epoch_val_precision_score = []
        self.epoch_val_recall_score = []
