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
            
            # Shuffle sequences and get original images
            shuffled_inputs, original_images = shuffle_sequence(inputs, device)
            
            optimizer.zero_grad()
            outputs = model(shuffled_inputs, task='sequence')
            
            # Calculate loss based on the difference between original images and outputs
            loss = criterion(outputs, original_images)
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
                shuffled_inputs, original_images = shuffle_sequence(inputs, device)
                outputs = model(shuffled_inputs, task='sequence')
                
                loss = criterion(outputs, original_images)
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
    """Shuffle columns within each sequence and return both shuffled sequence and original indices"""
    # Get dimensions
    batch_size, num_rows, num_cols = sequence.size()
    
    # Create random permutations for each sequence in the batch
    indices = torch.stack([torch.randperm(num_cols, device=device) for _ in range(batch_size)])
    
    # Create a range of indices for the batch dimension
    batch_indices = torch.arange(batch_size, device=device).view(-1, 1).expand(-1, num_cols)
    
    # Shuffle columns within each sequence
    shuffled_sequence = sequence[batch_indices, :, indices]
    
    return shuffled_sequence, sequence