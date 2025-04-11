import torch
import matplotlib.pyplot as plt
import json

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
        correct = 0
        total = 0

        with torch.no_grad():
            for inputs, targets in dataloader:
                inputs, targets = inputs.to(self.device), targets.to(self.device)
                outputs = self.model(inputs)
                loss = self.criterion(outputs, targets)
                total_loss += loss.item()
                
                _, predicted = torch.max(outputs, 1)
                correct += (predicted == targets).sum().item()
                total += targets.size(0)

        avg_loss = total_loss / len(dataloader)
        accuracy = 100 * correct / total
        return avg_loss, accuracy
    
    def save_log(self, log):
        fig, axes = plt.subplots(1, 2, figsize=(10, 5))

        axes[0].set_title(f'Loss')
        axes[0].set_xlabel(f'Epochs')
        axes[0].set_ylabel(f'Loss')
        axes[0].plot(log['train_loss'], label='Train')
        axes[0].plot(log['valid_loss'], label='Valid')
        axes[0].plot(log['test_loss'], label='Test')
        axes[0].legend()

        axes[1].set_title(f'Accuracy')
        axes[1].set_xlabel(f'Epochs')
        axes[1].set_ylabel(f'Acc')
        axes[1].plot(log['valid_acc'], label='Valid')
        axes[1].plot(log['test_acc'], label='Test')

        plt.tight_layout()
        plt.savefig(f'{self.save_dir}/training log.png', format='png')

        with open(f'{self.save_dir}/training log.json', 'w') as json_file:
            json.dump(log, json_file, indent=4)

    def run(self, num_epochs):
        best_test_loss = float("inf")
        best_test_acc = 0

        logging = {
            "train_loss": [],
            "valid_loss": [],
            "valid_acc": [],
            "test_loss": [],
            "test_acc": []
        }

        start_val_loss, start_accuracy = self.evaluate(self.test_loader)
        print(f"Epoch [{0}/{num_epochs}] - Val Loss: {start_val_loss:.4f}, Val Acc: {start_accuracy:.2f}%")

        for epoch in range(num_epochs):
            avg_train_loss = self.train()
            avg_val_loss, val_accuracy = self.evaluate(self.val_loader)
            avg_test_loss, test_accuracy = self.evaluate(self.test_loader)

            logging["train_loss"].append(avg_train_loss)
            logging["valid_loss"].append(avg_val_loss)
            logging["valid_acc"].append(val_accuracy)
            logging["test_loss"].append(avg_test_loss)
            logging["test_acc"].append(test_accuracy)

            print(f"Epoch [{epoch+1}/{num_epochs}] - Train Loss: {avg_train_loss:.4f}, Valid Loss: {avg_val_loss:.4f}, Valid Acc: {val_accuracy:.2f}%, Test Loss: {avg_test_loss:.4f}, Test Acc: {test_accuracy:.2f}%")

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