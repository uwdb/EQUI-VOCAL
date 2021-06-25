from utils import config
import itertools
import os
import torch

def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

def save_checkpoint(model, epoch, checkpoint_dir, stats):
    state = {
        'epoch': epoch,
        'state_dict': model.state_dict(),
        'stats': stats,
    }

    filename = os.path.join(checkpoint_dir,
        'epoch={}.checkpoint.pth.tar'.format(epoch))
    torch.save(state, filename)

def restore_checkpoint(model, checkpoint_dir, cuda=False, force=False,
    pretrain=False):
    """
    If a checkpoint exists, restores the PyTorch model from the checkpoint.
    Returns the model and the current epoch.
    """
    cp_files = [file_ for file_ in os.listdir(checkpoint_dir)
        if file_.startswith('epoch=') and file_.endswith('.checkpoint.pth.tar')]

    if not cp_files:
        print('No saved model parameters found')
        if force:
            raise Exception("Checkpoint not found")
        else:
            return model, 0, []

    # Find latest epoch
    for i in itertools.count(1):
        if 'epoch={}.checkpoint.pth.tar'.format(i) in cp_files:
            epoch = i
        else:
            break

    if not force:
        print("Which epoch to load from? Choose in range [0, {}]."
            .format(epoch), "Enter 0 to train from scratch.")
        print(">> ", end='')
        inp_epoch = int(input())
        if inp_epoch not in range(epoch+1):
            raise Exception("Invalid epoch number")
        if inp_epoch == 0:
            print("Checkpoint not loaded")
            clear_checkpoint(checkpoint_dir)
            return model, 0, []
    else:
        print("Which epoch to load from? Choose in range [1, {}].".format(epoch))
        inp_epoch = int(input())
        if inp_epoch not in range(1, epoch+1):
            raise Exception("Invalid epoch number")

    filename = os.path.join(checkpoint_dir,
        'epoch={}.checkpoint.pth.tar'.format(inp_epoch))

    print("Loading from checkpoint {}?".format(filename))

    if cuda:
        checkpoint = torch.load(filename)
    else:
        # Load GPU model on CPU
        checkpoint = torch.load(filename,
            map_location=lambda storage, loc: storage)

    try:
        start_epoch = checkpoint['epoch']
        stats = checkpoint['stats']
        if pretrain:
            model.load_state_dict(checkpoint['state_dict'], strict=False)
        else:
            model.load_state_dict(checkpoint['state_dict'])
        print("=> Successfully restored checkpoint (trained for {} epochs)"
            .format(checkpoint['epoch']))
    except:
        print("=> Checkpoint not successfully restored")
        raise

    return model, inp_epoch, stats

def clear_checkpoint(checkpoint_dir):
    filelist = [ f for f in os.listdir(checkpoint_dir) if f.endswith(".pth.tar") ]
    for f in filelist:
        os.remove(os.path.join(checkpoint_dir, f))

    print("Checkpoint successfully removed")

def predictions(logits):
    """
    Given the network output, determines the predicted class index

    Returns:
        the predicted class output as a PyTorch Tensor
    """
    result = torch.argmax(logits, dim=1)
    return result

def topk_pred(logits, k=2):
    """
    Given the network output, determines the topk predicted class index and prob

    Returns:
        the topk predicted class output as a PyTorch Tensor of namedtuples
    """
    # apply softmax normalization
    prob_dist = torch.nn.functional.softmax(logits, dim=1)
    result = torch.topk(prob_dist, k, dim=1)
    return result