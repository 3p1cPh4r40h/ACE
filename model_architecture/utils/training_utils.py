import torch
import torch.nn as nn
import torch.optim as optim

def train(model, criterion, optimizer, train_dataloader, val_dataloader, epochs=1000, loss_hit_epochs=50, early_stop_epochs=200, device='cpu'):    
    worse_loss = 0
    early_stop = 0
    best_loss = float('inf')
    best_weights = None
    losses = []
    val_losses = []

    for epoch in range(epochs):
        model.train()
        running_loss = 0.0

        for inputs, labels in train_dataloader:
            inputs, labels = inputs.to(device), labels.to(device)
            
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()

        # Validation step
        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for inputs, labels in val_dataloader:
                inputs, labels = inputs.to(device), labels.to(device)
                outputs = model(inputs)
                loss = criterion(outputs, labels)
                val_loss += loss.item()

        current_loss = running_loss / len(train_dataloader)
        current_val_loss = val_loss / len(val_dataloader)
        losses.append(current_loss)
        val_losses.append(current_val_loss)
        print(f"Epoch {epoch+1}, Training Loss: {current_loss}, Validation Loss: {current_val_loss}")

        if current_val_loss < best_loss:
            best_loss = current_val_loss
            best_weights = model.state_dict()
            worse_loss = 0
            early_stop = 0
        else:
            worse_loss += 1
            early_stop += 1

        if worse_loss >= loss_hit_epochs-1:
            print('Weight Optimization Hit')
            for g in optimizer.param_groups:
                g['lr'] = g['lr'] * 0.5
            worse_loss = 0

        if early_stop >= early_stop_epochs-1:
            print('Ending Training Early')
            break
    
    return [best_weights, losses, val_losses]

def train_sequence_ordering(model, criterion, optimizer, train_dataloader, val_dataloader, epochs=1000, loss_hit_epochs=50, early_stop_epochs=200, device='cpu'):
    worse_loss = 0
    early_stop = 0
    best_loss = float('inf')
    best_weights = None
    losses = []
    val_losses = []

    for epoch in range(epochs):
        model.train()
        running_loss = 0.0

        for inputs, _ in train_dataloader:
            inputs = inputs.to(device)
            batch_size = inputs.size(0)
            
            # Shuffle sequences and get original indices
            shuffled_inputs, original_indices = shuffle_sequence(inputs, device)
            
            optimizer.zero_grad()
            outputs = model(shuffled_inputs, task='sequence')
            
            # Ensure indices are properly formatted for CrossEntropyLoss
            original_indices = original_indices % 9  # Ensure indices are in range [0,8]
            
            loss = criterion(outputs, original_indices)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()

        # Validation step
        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for inputs, _ in val_dataloader:
                inputs = inputs.to(device)
                batch_size = inputs.size(0)
                shuffled_inputs, original_indices = shuffle_sequence(inputs, device)
                outputs = model(shuffled_inputs, task='sequence')
                
                original_indices = original_indices % 9  # Ensure indices are in range [0,8]
                
                loss = criterion(outputs, original_indices)
                val_loss += loss.item()

        current_loss = running_loss / len(train_dataloader)
        current_val_loss = val_loss / len(val_dataloader)
        losses.append(current_loss)
        val_losses.append(current_val_loss)
        print(f"Sequence Ordering Epoch {epoch+1}, Training Loss: {current_loss}, Validation Loss: {current_val_loss}")

        if current_val_loss < best_loss:
            best_loss = current_val_loss
            best_weights = model.state_dict()
            worse_loss = 0
            early_stop = 0
        else:
            worse_loss += 1
            early_stop += 1

        if worse_loss >= loss_hit_epochs-1:
            print('Weight Optimization Hit')
            for g in optimizer.param_groups:
                g['lr'] = g['lr'] * 0.5
            worse_loss = 0

        if early_stop >= early_stop_epochs-1:
            print('Ending Sequence Ordering Training Early')
            break

    return [best_weights, losses, val_losses]

def shuffle_sequence(sequence, device):
    """Shuffle a sequence and return both shuffled sequence and original indices"""
    batch_size = sequence.size(0)
    indices = torch.randperm(batch_size, device=device)
    return sequence[indices], indices 