from jesnk_utils.telebot import Telebot

import torch
import torchvision
import torchvision.transforms as transforms
import matplotlib.pyplot as plt


if __name__ == '__main__':
    telebot = Telebot()

    # get args
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--epoch', type=str, default=None)

    args = parser.parse_args()
    print(args)
    telebot.send_message('vit.py: args: ' + str(args))
    
    # 데이터 준비
    transform = transforms.Compose([
        transforms.Resize(224),  # ViT 모델은 224x224 입력을 기대합니다.
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])

    train_dataset = torchvision.datasets.CIFAR10(root='./data', train=True, download=True, transform=transform)
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=64, shuffle=True)

    test_dataset = torchvision.datasets.CIFAR10(root='./data', train=False, download=True, transform=transform)
    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=64, shuffle=False)

    # Vision Transformer 모델
    #model = torchvision.models.vision_transformer.vit_small_patch16_224(pretrained=False, num_classes=10)
    model = torchvision.models.vision_transformer.VisionTransformer(image_size=224, patch_size=16, num_classes=10, num_layers=6,num_heads=6, mlp_dim=512, hidden_dim=768)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    criterion = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

    telebot.send_message('vit.py: Start training')
    # 학습
    num_epochs = int(args.epoch)
    for epoch in range(num_epochs):
        model.train()
        for images, labels in train_loader:
            images, labels = images.to(device), labels.to(device)

            outputs = model(images)
            loss = criterion(outputs, labels)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        print(f"Epoch [{epoch + 1}/{num_epochs}], Loss: {loss.item():.4f}")
        telebot.send_message(f"Epoch [{epoch + 1}/{num_epochs}], Loss: {loss.item():.4f}")


    telebot.send_message('vit.py: Start testing')
    # 결과 확인
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for images, labels in test_loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            _, predicted = outputs.max(1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    print(f"Accuracy: {100 * correct / total:.2f}%")
    telebot.send_message(f"Accuracy: {100 * correct / total:.2f}%")

    # 결과 시각화
    def imshow(img):
        img = img / 2 + 0.5  # unnormalize
        plt.imshow(img.permute(1, 2, 0))
        plt.show()
        # save png
        plt.savefig('vit_test.png')
        telebot.send_image('./vit_test.png')

    dataiter = iter(test_loader)
    images, labels = dataiter.next()

    imshow(torchvision.utils.make_grid(images[:4]))

    print('GroundTruth: ', ' '.join('%5s' % train_dataset.classes[labels[j]] for j in range(4)))
    telebot.send_message('GroundTruth: ' + ' '.join('%5s' % train_dataset.classes[labels[j]] for j in range(4)))

    outputs = model(images[:4].to(device))
    _, predicted = outputs.max(1)
    print('Predicted: ', ' '.join('%5s' % train_dataset.classes[predicted[j]] for j in range(4)))
    telebot.send_message('Predicted: ' + ' '.join('%5s' % train_dataset.classes[predicted[j]] for j in range(4)))

    # save model
    torch.save(model.state_dict(), 'vit_model.pt')
    telebot.send_file('./vit_model.pt saved')

    telebot.send_message('vit.py: End')
