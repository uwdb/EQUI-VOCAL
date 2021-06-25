from utils import config
import itertools
import os
import torch

def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

def save_checkpoint(model, epoch, best_loss, filename):
    """Save checkpoint if a new best is achieved"""
    print ("=> Saving a new best")
    state = {
        'epoch': epoch,
        'model': model,
        'best_loss': best_loss
    }
    torch.save(state, filename) # save checkpoint

def restore_checkpoint(model, filename, cuda=False):
    """
    If a checkpoint exists, restores the PyTorch model from the checkpoint.
    Returns the model and the current epoch.
    """

    if cuda:
        checkpoint = torch.load(filename)
    else:
        # Load GPU model on CPU
        checkpoint = torch.load(filename,
            map_location=lambda storage, loc: storage)

    try:
        start_epoch = checkpoint['epoch']
        best_loss = checkpoint['best_loss']
        model = checkpoint['model']
        print("=> Successfully restored checkpoint (trained for {} epochs)"
            .format(checkpoint['epoch']))
    except:
        print("=> Checkpoint not successfully restored")
        raise

    return model

def clear_checkpoint(checkpoint_dir):
    filelist = [ f for f in os.listdir(checkpoint_dir) if f.endswith(".pth.tar") ]
    for f in filelist:
        os.remove(os.path.join(checkpoint_dir, f))

    print("Checkpoint successfully removed")

def restore_model(model_dir):
    """
    If a model exists, restores the PyTorch model.
    Returns the model and the current epoch.
    """
    cp_files = [file_ for file_ in os.listdir(model_dir)
        if file_.startswith('epoch=') and file_.endswith('.model.pth.tar')]

    if not cp_files:
        raise Exception("No saved model found")

    # Find latest epoch
    for i in itertools.count(1):
        if 'epoch={}.model.pth.tar'.format(i) in cp_files:
            epoch = i
        else:
            break

    print("Which epoch to load from? Choose in range [1, {}]."
        .format(epoch), "Don't enter 0.")
    print(">> ", end='')
    inp_epoch = int(input())
    if inp_epoch not in range(1, epoch+1):
        raise Exception("Invalid epoch number")

    filename = os.path.join(model_dir,
        'epoch={}.model.pth.tar'.format(inp_epoch))

    print("Loading from checkpoint {}?".format(filename))

    try:
        model = torch.load(filename)
        print("=> Successfully restored model (trained for {} epochs)"
            .format(inp_epoch))
    except:
        print("=> Checkpoint not successfully restored")
        raise

    return model, inp_epoch

def predictions(logits):
    """
    Given the network output, determines the predicted class index

    Returns:
        the predicted class output as a PyTorch Tensor
    """
    # TODO: implement this function
    # result = torch.argmax(logits, dim=1).type(torch.FloatTensor)
    result = torch.argmax(logits, dim=1)
    return result
    #
