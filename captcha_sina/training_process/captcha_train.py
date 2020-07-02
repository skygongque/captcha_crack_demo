import torch
# 引入模型
from captcha_model import Model
# 引入自己的数据集
from sinadataset import SinaDataset,CaptchaDataset
from torch.utils.data import DataLoader,Dataset
import string
import tqdm


# 相关的参数
TRAIN_DATASET_PATH = r'captcha_sina\training_process\dataset\train'
characters = '-' + string.digits + string.ascii_lowercase
width, height, n_len, n_classes = 100, 40, 6, len(characters)
n_input_length = 12
print(characters, width, height, n_len, n_classes)
# Device configuration
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# 测试模型输出
model = Model(n_classes, input_shape=(3, height, width))
inputs = torch.zeros((32, 3, height, width))
outputs = model(inputs)
print(outputs.shape)


batch_size = 12
# trans_set的length调一下
# train_set = CaptchaDataset(characters = characters, width=width, height=height, input_length=n_input_length, label_length=n_len,folder=TRAIN_DATASET_PATH)
train_set = CaptchaDataset(characters, 10 * batch_size, width, height, n_input_length, n_len)
# valid_set = CaptchaDataset(characters, 100 * batch_size, width, height, n_input_length, n_len)
train_loader = DataLoader(train_set, batch_size=batch_size, num_workers=1,shuffle=True,drop_last=True)
# valid_loader = DataLoader(valid_set, batch_size=batch_size, num_workers=12)

# test_loader  = iter(train_loader)
# image, target, input_length, label_length = test_loader.next()
# print(target, input_length, label_length)
# print(type(train_loader),type(test_loader))
# dataset = SinaDataset(characters, width, height, n_input_length, n_len,TRAIN_DATASET_PATH)
# print('dataset.length',dataset.length,'dataset.label_length',dataset.label_length)
# image, target, input_length, label_length = dataset[0]
# print(''.join([characters[x] for x in target]), input_length, label_length)
# to_pil_image(image).show()


model = Model(n_classes, input_shape=(3, height, width))
# gpu加速 如果可用的话
model.to(device)

def decode(sequence):
    a = ''.join([characters[x] for x in sequence])
    s = ''.join([x for j, x in enumerate(a[:-1]) if x != characters[0] and x != a[j+1]])
    if len(s) == 0:
        return ''
    if a[-1] != characters[0] and s[-1] != a[-1]:
        s += a[-1]
    return s

def decode_target(sequence):
    return ''.join([characters[x] for x in sequence]).replace(' ', '')

def calc_acc(target, output):
    output_argmax = output.detach().permute(1, 0, 2).argmax(dim=-1)
    target = target.cpu().numpy()
    output_argmax = output_argmax.cpu().numpy()
    a = np.array([decode_target(true) == decode(pred) for true, pred in zip(target, output_argmax)])
    return a.mean()



def train(model, optimizer, epoch, dataloader):
    model.train()
    loss_mean = 0
    acc_mean = 0
    with tqdm(dataloader) as pbar:
        for batch_index, (data, target, input_lengths, target_lengths) in enumerate(pbar):
            data, target = data.to(device), target.to(device)
            
            optimizer.zero_grad()
            output = model(data)
            
            output_log_softmax = F.log_softmax(output, dim=-1)
            loss = F.ctc_loss(output_log_softmax, target, input_lengths, target_lengths)
            
            loss.backward()
            optimizer.step()

            loss = loss.item()
            acc = calc_acc(target, output)
            
            if batch_index == 0:
                loss_mean = loss
                acc_mean = acc
            
            loss_mean = 0.1 * loss + 0.9 * loss_mean
            acc_mean = 0.1 * acc + 0.9 * acc_mean
            
            pbar.set_description(f'Epoch: {epoch} Loss: {loss_mean:.4f} Acc: {acc_mean:.4f} ')

def valid(model, optimizer, epoch, dataloader):
    model.eval()
    with tqdm(dataloader) as pbar, torch.no_grad():
        loss_sum = 0
        acc_sum = 0
        for batch_index, (data, target, input_lengths, target_lengths) in enumerate(pbar):
            data, target = data.to(device), target.to(device)
            
            output = model(data)
            output_log_softmax = F.log_softmax(output, dim=-1)
            loss = F.ctc_loss(output_log_softmax, target, input_lengths, target_lengths)
            
            loss = loss.item()
            acc = calc_acc(target, output)
            
            loss_sum += loss
            acc_sum += acc
            
            loss_mean = loss_sum / (batch_index + 1)
            acc_mean = acc_sum / (batch_index + 1)
            
            pbar.set_description(f'Test : {epoch} Loss: {loss_mean:.4f} Acc: {acc_mean:.4f} ')


optimizer = torch.optim.Adam(model.parameters(), 1e-3, amsgrad=True)
epochs = 10
for epoch in range(1, epochs + 1):
    train(model, optimizer, epoch, train_loader)
    # valid(model, optimizer, epoch, valid_loader)



optimizer = torch.optim.Adam(model.parameters(), 1e-4, amsgrad=True)
epochs = 5
for epoch in range(1, epochs + 1):
    train(model, optimizer, epoch, train_loader)
    # valid(model, optimizer, epoch, valid_loader)


# 保存模型
torch.save(model.state_dict(), 'ctc_702.pth')
pritn('saved model')