import torch
import matplotlib.pyplot as plt
import json
from sklearn.metrics import precision_score, recall_score, f1_score, accuracy_score

class CNN_Trainer:
    def __init__(self, save_dir, model, train_loader, val_loader, test_loader, criterion, optimizer, device):
        self.save_dir = save_dir
        self.model = model.to(device)
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.test_loader = test_loader
        self.criterion = criterion
        self.optimizer = optimizer
        self.device = device

    def train(self):
        self.model.train()
        total_loss = 0

        for inputs, targets in self.train_loader:
            inputs, targets = inputs.to(self.device), targets.to(self.device)

            self.optimizer.zero_grad()
            outputs = self.model(inputs)
            loss = self.criterion(outputs, targets)
            loss.backward()

            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
            self.optimizer.step()

            for p in self.model.classifier.parameters():
                p.data.clamp_(-1, 1)

            total_loss += loss.item()

        avg_train_loss = total_loss / len(self.train_loader)

        return avg_train_loss

    def evaluate(self, dataloader):
        self.model.eval()
        total_loss = 0

        all_targets = []
        all_predictions = []

        with torch.no_grad():
            for inputs, targets in dataloader:
                inputs, targets = inputs.to(self.device), targets.to(self.device)
                outputs = self.model(inputs)
                loss = self.criterion(outputs, targets)
                total_loss += loss.item()

                _, predicted = torch.max(outputs, 1)
                all_targets.extend(targets.cpu().numpy())
                all_predictions.extend(predicted.cpu().numpy())

        avg_loss = total_loss / len(dataloader)

        accuracy = accuracy_score(all_targets, all_predictions) * 100
        precision = precision_score(all_targets, all_predictions, average='macro', zero_division=0)
        recall = recall_score(all_targets, all_predictions, average='macro', zero_division=0)
        f1 = f1_score(all_targets, all_predictions, average='macro', zero_division=0)

        return avg_loss, accuracy, precision, recall, f1
    
    def save_log(self, log):
        fig, axes = plt.subplots(2, 3, figsize=(18, 10))
        fig.suptitle('Training Progress', fontsize=22)

        # --- Loss ---
        ax = axes[0, 0]
        ax.set_title('Loss', fontsize=16)
        ax.set_xlabel('Epochs', fontsize=14)
        ax.set_ylabel('Loss', fontsize=14)
        ax.grid(True, linestyle='--', alpha=0.7)
        ax.plot(log['train_loss'], label='Train')
        ax.plot(log['valid_loss'], label='Valid')
        ax.plot(log['test_loss'], label='Test')
        ax.legend(fontsize=12)
        ax.tick_params(axis='both', which='major', labelsize=12)

        # --- Accuracy ---
        ax = axes[0, 1]
        ax.set_title('Accuracy', fontsize=16)
        ax.set_xlabel('Epochs', fontsize=14)
        ax.set_ylabel('Accuracy (%)', fontsize=14)
        ax.grid(True, linestyle='--', alpha=0.7)
        ax.plot(log['valid_acc'], label='Valid')
        ax.plot(log['test_acc'], label='Test')
        ax.legend(fontsize=12)
        ax.tick_params(axis='both', which='major', labelsize=12)

        # --- Precision ---
        ax = axes[1, 0]
        ax.set_title('Precision', fontsize=16)
        ax.set_xlabel('Epochs', fontsize=14)
        ax.set_ylabel('Precision', fontsize=14)
        ax.grid(True, linestyle='--', alpha=0.7)
        ax.plot(log['valid_precision'], label='Valid')
        ax.plot(log['test_precision'], label='Test')
        ax.legend(fontsize=12)
        ax.tick_params(axis='both', which='major', labelsize=12)

        # --- Recall ---
        ax = axes[1, 1]
        ax.set_title('Recall', fontsize=16)
        ax.set_xlabel('Epochs', fontsize=14)
        ax.set_ylabel('Recall', fontsize=14)
        ax.grid(True, linestyle='--', alpha=0.7)
        ax.plot(log['valid_recall'], label='Valid')
        ax.plot(log['test_recall'], label='Test')
        ax.legend(fontsize=12)
        ax.tick_params(axis='both', which='major', labelsize=12)

        # --- F1 ---
        ax = axes[1, 2]
        ax.set_title('F1 Score', fontsize=16)
        ax.set_xlabel('Epochs', fontsize=14)
        ax.set_ylabel('F1', fontsize=14)
        ax.grid(True, linestyle='--', alpha=0.7)
        ax.plot(log['valid_f1'], label='Valid')
        ax.plot(log['test_f1'], label='Test')
        ax.legend(fontsize=12)
        ax.tick_params(axis='both', which='major', labelsize=12)

        # --- Empty (bottom right) ---
        fig.delaxes(axes[0, 2])

        plt.tight_layout(rect=[0, 0, 1, 0.96])
        plt.savefig(f'{self.save_dir}/training_log.png', format='png')
        plt.close()

        # Save JSON
        with open(f'{self.save_dir}/training_log.json', 'w') as json_file:
            json.dump(log, json_file, indent=4)

    def run(self, num_epochs):
        best_test_loss = float("inf")
        best_test_acc = 0

        logging = {
            "train_loss": [],
            "valid_loss": [],
            "valid_acc": [],
            "valid_precision": [],
            "valid_recall": [],
            "valid_f1": [],
            "test_loss": [],
            "test_acc": [],
            "test_precision": [],
            "test_recall": [],
            "test_f1": []
        }

        start_val_loss, start_accuracy, _, _, _ = self.evaluate(self.test_loader)
        print(f"Epoch [{0}/{num_epochs}] - Val Loss: {start_val_loss:.4f}, Val Acc: {start_accuracy:.2f}%")

        for epoch in range(num_epochs):
            avg_train_loss = self.train()
            avg_val_loss, val_accuracy, val_precision, val_recall, val_f1 = self.evaluate(self.val_loader)
            avg_test_loss, test_accuracy, test_precision, test_recall, test_f1 = self.evaluate(self.test_loader)

            logging["train_loss"].append(avg_train_loss)
            logging["valid_loss"].append(avg_val_loss)
            logging["valid_acc"].append(val_accuracy)
            logging["valid_precision"].append(val_precision)
            logging["valid_recall"].append(val_recall)
            logging["valid_f1"].append(val_f1)
            logging["test_loss"].append(avg_test_loss)
            logging["test_acc"].append(test_accuracy)
            logging["test_precision"].append(test_precision)
            logging["test_recall"].append(test_recall)
            logging["test_f1"].append(test_f1)

            print(f"Epoch [{epoch+1}/{num_epochs}] - "
                f"Train Loss: {avg_train_loss:.4f}, "
                f"Valid Loss: {avg_val_loss:.4f}, "
                f"Valid Acc: {val_accuracy:.2f}%, "
                f"Test Loss: {avg_test_loss:.4f}, "
                f"Test Acc: {test_accuracy:.2f}%, "
                f"Test Precision: {test_precision:.4f}, "
                f"Test Recall: {test_recall:.4f}, "
                f"Test F1: {test_f1:.4f}")

            # save model
            if avg_test_loss < best_test_loss:
                best_test_loss = avg_test_loss
                torch.save(self.model.state_dict(), f"{self.save_dir}/best_model_loss.pth")
                print(f"Best loss model saved at epoch {epoch+1}")
            if test_accuracy > best_test_acc:
                best_test_acc = test_accuracy
                torch.save(self.model.state_dict(), f"{self.save_dir}/best_model_accuracy.pth")
                print(f"Best accuracy model saved at epoch {epoch+1}")

        torch.save(self.model.state_dict(), f"{self.save_dir}/Final_model.pth")
        print(f"Final model saved at epoch {epoch+1}")

        # save log
        self.save_log(logging)