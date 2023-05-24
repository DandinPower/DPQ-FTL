import torch

def convert_checkpoint(input_path, output_path):
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    checkpoint = torch.load(input_path, map_location={'cuda:1': 'cuda:0'})
    torch.save(checkpoint, output_path)

for i in range(100):
    path = f'dpq/history_weight/scratch_wdev_0/scratch_wdev_0_{i}.pth'
    convert_checkpoint(path, path)

path = f'dpq/history_weight/scratch_wdev_0/scratch_wdev_0_finish.pth'
convert_checkpoint(path, path)