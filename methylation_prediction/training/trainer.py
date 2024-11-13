# training/trainer.py

import torch
import logging
from tqdm import tqdm

class Trainer:
    def __init__(self, model, criterion, optimizer, scheduler, device, config):
        self.model = model
        self.criterion = criterion
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.device = device
        self.config = config
        self.num_epochs = config.get('training', {}).get('num_epochs', 10)  # Default to 10 if missing
        self.model_save_path = config.get('training', {}).get('model_save_path', None)
        self.train_losses = []
        self.val_losses = []

    def train(self, train_loader, val_loader):
        best_val_loss = float('inf')
        num_epochs = self.num_epochs
        model_save_path = self.model_save_path

        for epoch in range(num_epochs):
            self.model.train()
            running_loss = 0.0
            for batch in tqdm(train_loader, desc=f"Epoch {epoch+1}/{num_epochs}"):
                sequences = batch['sequence'].to(self.device)
                features = batch['features'].to(self.device)
                labels = batch['label'].to(self.device)
                sequence_lengths = batch['sequence_length'].to(self.device)

                self.optimizer.zero_grad()
                outputs = self.model(sequences, sequence_lengths, features)
                loss = self.criterion(outputs, labels)
                loss.backward()
                self.optimizer.step()

                running_loss += loss.item() * sequences.size(0)

            epoch_loss = running_loss / len(train_loader.dataset)
            self.train_losses.append(epoch_loss)

            # Validation
            val_loss = self.validate(val_loader)
            self.val_losses.append(val_loss)

            # Step the scheduler
            self.scheduler.step(val_loss)

            logging.info(f'Epoch {epoch+1}/{num_epochs}, Training Loss: {epoch_loss:.6f}, Validation Loss: {val_loss:.6f}')

            # Save the best model
            if model_save_path and val_loss < best_val_loss:
                best_val_loss = val_loss
                torch.save(self.model.state_dict(), model_save_path)
                logging.info(f"Epoch {epoch+1}: New best model saved with validation loss {val_loss:.6f}")

    def validate(self, val_loader):
        self.model.eval()
        val_running_loss = 0.0
        with torch.no_grad():
            for batch in val_loader:
                sequences = batch['sequence'].to(self.device)
                features = batch['features'].to(self.device)
                labels = batch['label'].to(self.device)
                sequence_lengths = batch['sequence_length'].to(self.device)

                outputs = self.model(sequences, sequence_lengths, features)
                loss = self.criterion(outputs, labels)
                val_running_loss += loss.item() * sequences.size(0)

        avg_val_loss = val_running_loss / len(val_loader.dataset)
        return avg_val_loss

    def get_predictions(self, loader):
        self.model.eval()
        all_predictions = []
        all_labels = []
        with torch.no_grad():
            for batch in loader:
                sequences = batch['sequence'].to(self.device)
                features = batch['features'].to(self.device)
                labels = batch['label'].to(self.device)
                sequence_lengths = batch['sequence_length'].to(self.device)

                outputs = self.model(sequences, sequence_lengths, features)
                all_predictions.extend(outputs.cpu().numpy())
                all_labels.extend(labels.cpu().numpy())
        return all_predictions, all_labels
