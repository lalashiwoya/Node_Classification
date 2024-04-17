import torch


def predict(node_embs, adj, labels, idx, model, device):
    model.eval()
    with torch.no_grad():
        idx = torch.LongTensor(idx).to(device)
        output = model(node_embs, adj).to(device)
        preds = output[idx].max(1)[1].type_as(labels[idx])
    return preds.cpu().numpy()

