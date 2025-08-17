# 训练+测试
import torch
import torch.nn as nn
import torch.utils.data as Data
import torchvision
import cv2
import argparse
from torchvision import transforms

# 定义CNN模型
class CNN(nn.Module):
    def __init__(self):
        super(CNN, self).__init__()
        self.conv1 = nn.Sequential(
            nn.Conv2d(in_channels=1, out_channels=16, kernel_size=5, stride=1, padding=2),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2),
        )
        self.conv2 = nn.Sequential(
            nn.Conv2d(16, 32, 5, 1, 2),
            nn.ReLU(),
            nn.MaxPool2d(2),
        )
        self.out = nn.Linear(32 * 7 * 7, 10)

    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = x.view(x.size(0), -1)
        output = self.out(x)
        return output

# 训练模型
def train(model, train_loader, test_x, test_y, epoch_num=1, lr=0.001):
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    loss_func = nn.CrossEntropyLoss()

    for epoch in range(epoch_num):
        for step, (b_x, b_y) in enumerate(train_loader):
            output = model(b_x)
            loss = loss_func(output, b_y)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            if step % 100 == 0:
                test_output = model(test_x)
                pred_y = torch.max(test_output, 1)[1].data.numpy()
                accuracy = float((pred_y == test_y.data.numpy()).astype(int).sum()) / float(test_y.size(0))
                print(f'Epoch: {epoch} | train loss: {loss.data.numpy():.4f} | test accuracy: {accuracy:.2f}')

    torch.save(model.state_dict(), 'cnn2.pkl')
    print("模型已保存到 cnn2.pkl")

# 在测试集上预测
def predict_on_test_set(model, test_x, test_y):
    model.load_state_dict(torch.load('cnn2.pkl'))
    model.eval()
    
    inputs = test_x[:32]
    test_output = model(inputs)
    pred_y = torch.max(test_output, 1)[1].data.numpy()
    
    print(f"被测数字: {test_y[:32].numpy()}")
    print(f"测试结果: {pred_y}")

    img = torchvision.utils.make_grid(inputs)
    img = img.numpy().transpose(1, 2, 0)
    cv2.imshow('Test Images', img)
    cv2.waitKey(0)

# 预测单个图片
def predict_image(model, image_path):
    model.load_state_dict(torch.load('cnn2.pkl'))
    model.eval()

    # 图像预处理
    transform = transforms.ToTensor()

    # 读取并处理图像
    img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    if img is None:
        print(f"无法读取图片: {image_path}")
        return
        
    img = cv2.resize(img, (28, 28))
    img_tensor = transform(img).unsqueeze(0)

    # 预测
    with torch.no_grad():
        output = model(img_tensor)
        pred = torch.max(output, 1)[1].data.numpy()
        print(f"图片 '{image_path}' 的预测结果是: {pred[0]}")

    # 显示图片
    cv2.imshow(f'Predicted: {pred[0]}', img)
    cv2.waitKey(0)


def main():
    parser = argparse.ArgumentParser(description='MNIST CNN - 训练和预测')
    parser.add_argument('--mode', type=str, default='predict', choices=['train', 'predict', 'predict_test'],
                        help='选择模式: train (训练), predict (预测单张图片), or predict_test (在测试集上预测)')
    parser.add_argument('--image_path', type=str, help='需要预测的图片路径')
    args = parser.parse_args()

    # 加载数据
    train_data = torchvision.datasets.MNIST(root='./data/', train=True, transform=torchvision.transforms.ToTensor(), download=True)
    test_data = torchvision.datasets.MNIST(root='./data/', train=False)
    
    train_loader = Data.DataLoader(dataset=train_data, batch_size=50, shuffle=True)
    test_x = torch.unsqueeze(test_data.data, dim=1).type(torch.FloatTensor)[:2000] / 255
    test_y = test_data.targets[:2000]

    cnn = CNN()

    if args.mode == 'train':
        train(cnn, train_loader, test_x, test_y, epoch_num=2)
    elif args.mode == 'predict':
        if not args.image_path:
            print("错误: 使用 'predict' 模式时必须提供 --image_path 参数。")
            return
        predict_image(cnn, args.image_path)
    elif args.mode == 'predict_test':
        predict_on_test_set(cnn, test_x, test_y)

if __name__ == '__main__':
    main()