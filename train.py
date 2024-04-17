
import torch


def create_batches(indices, batch_size):
    num_batches = len(indices) // batch_size
    if len(indices) % batch_size != 0:
        num_batches += 1

    for i in range(num_batches):
        start_idx = i * batch_size
        end_idx = min((i + 1) * batch_size, len(indices))
        yield indices[start_idx:end_idx]

def train_epoch(node_embs, adj, labels, idx, model, optimizer, loss_func, batch_size, device):
    model.train()
    for idx in create_batches(idx, batch_size):
        idx = torch.LongTensor(idx).to(device)
        optimizer.zero_grad()
        output = model(node_embs, adj).to(device)
        loss = loss_func(output[idx], labels[idx])
        loss.backward()
        optimizer.step()
         
    

def train(node_embs, adj, labels, train_idx, test_idx, model, optimizer, loss_func, num_epochs, batch_size, device, print_loss_interval = 100):
    for epoch in range(num_epochs):
        train_epoch(node_embs, adj, labels, train_idx, model, optimizer, loss_func, batch_size, device)
        
        if(epoch+1)%print_loss_interval == 0 or epoch == 0:
            train_loss, train_acc = compute_loss_accuracy(node_embs, adj, labels, train_idx, model, loss_func, device)
            test_loss, test_acc = compute_loss_accuracy(node_embs, adj, labels, test_idx, model, loss_func, device)
            print('Epochs: {}, Train Loss: {:.3f}, Train Acc: {:.3f}, Test Loss: {:.3f}, Test Acc: {:.3f}'.format(epoch, 
                                                                                                                train_loss, 
                                                                                                                train_acc, 
                                                                                                                test_loss, 
                                                                                                                test_acc))
        

def compute_loss_accuracy(node_embs, adj, labels, idx, model, loss_func, device):
     model.eval()
     total_loss = 0
     num_batches = 0
     correct = 0
     with torch.no_grad():
        
        idx = torch.LongTensor(idx).to(device)
        output = model(node_embs, adj).to(device)
        loss = loss_func(output[idx], labels[idx])
        total_loss += loss.item()
        preds = output[idx].max(1)[1].type_as(labels[idx])
        correct += preds.eq(labels[idx]).double().sum()
        num_batches += 1
    
        acc = correct / len(idx)
        return loss, acc
        
            
         