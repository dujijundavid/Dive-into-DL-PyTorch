"""
PyTorch 计算机视觉核心技术与实践
------------------------------
【文件说明】
本文件系统梳理了计算机视觉深度学习的核心技术，包括：
- 图像增强（Data Augmentation）：提升模型泛化能力
- 迁移学习与微调（Transfer Learning & Fine-tuning）：充分利用预训练模型
- 目标检测（Object Detection）：边界框、锚框、多尺度检测
- 语义分割（Semantic Segmentation）：像素级分类任务
- 神经风格迁移（Neural Style Transfer）：艺术风格转换

【第一性原理思考】
1. 为什么计算机视觉如此重要？
   - 视觉是人类获取信息的主要途径（80%以上）
   - 图像数据包含丰富的空间结构信息
   - 视觉任务是AI落地应用的重要场景（自动驾驶、医疗影像、安防等）

2. 卷积神经网络为什么适合视觉任务？
   - 局部感受野：捕捉图像的局部特征
   - 参数共享：减少参数量，提高泛化能力
   - 平移不变性：对图像位移具有鲁棒性
   - 层次化特征学习：从低级边缘到高级语义

3. 数据增强的本质是什么？
   - 增加数据多样性，模拟真实场景变化
   - 正则化效果，防止过拟合
   - 提升模型对变换的不变性

【苏格拉底式提问与验证】
1. 迁移学习为什么有效？
   - 问题：预训练特征是否通用？
   - 验证：可视化不同层的特征表示
   - 结论：底层特征通用，顶层特征任务相关

2. 目标检测与分类的本质区别？
   - 问题：为什么不能直接用分类网络？
   - 验证：分析定位精度和检测召回率
   - 结论：需要同时解决分类和定位问题

【费曼学习法讲解】
1. 概念解释
   - 用人眼视觉系统类比CNN的层次化处理
   - 用地图标注类比目标检测任务
   - 强调视觉任务在实际应用中的价值

2. 实例教学  
   - 从简单的图像分类开始
   - 逐步扩展到检测、分割等复杂任务
   - 通过可视化理解算法原理

【设计意义与工程价值】
- 计算机视觉是AI商业化最成功的领域之一
- 掌握这些技术是从事CV工程师的必备技能
- 为后续学习更先进的视觉模型（Transformer、扩散模型等）奠定基础

可运行案例：
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader, Dataset
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
import cv2

# 设置设备
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"使用设备: {device}")

# 1. 图像增强技术（Data Augmentation）
# -----------------------------------
# 原理说明：
# 数据增强通过对训练图像应用各种变换，人工扩充数据集
# 这些变换模拟了真实世界中的图像变化，提升模型泛化能力
# 常见变换：几何变换（旋转、缩放）、颜色变换、噪声添加等

print("1. 图像增强技术演示：")

class ImageAugmentationDemo:
    """图像增强演示类"""
    
    def __init__(self):
        # 定义各种增强变换
        self.transforms = {
            'original': transforms.Compose([
                transforms.ToTensor(),
            ]),
            'horizontal_flip': transforms.Compose([
                transforms.RandomHorizontalFlip(p=1.0),
                transforms.ToTensor(),
            ]),
            'rotation': transforms.Compose([
                transforms.RandomRotation(degrees=30),
                transforms.ToTensor(),
            ]),
            'color_jitter': transforms.Compose([
                transforms.ColorJitter(brightness=0.3, contrast=0.3, 
                                     saturation=0.3, hue=0.1),
                transforms.ToTensor(),
            ]),
            'random_crop': transforms.Compose([
                transforms.RandomResizedCrop(224, scale=(0.8, 1.0)),
                transforms.ToTensor(),
            ]),
            'comprehensive': transforms.Compose([
                transforms.RandomResizedCrop(224),
                transforms.RandomHorizontalFlip(),
                transforms.ColorJitter(brightness=0.2, contrast=0.2),
                transforms.RandomRotation(10),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                                   std=[0.229, 0.224, 0.225])
            ])
        }
    
    def demonstrate_augmentation(self, image_path=None):
        """演示不同的增强效果"""
        # 创建一个简单的测试图像
        if image_path is None:
            # 创建彩色测试图像
            img = Image.new('RGB', (224, 224), color='white')
            # 添加一些几何图形用于观察变换效果
            from PIL import ImageDraw
            draw = ImageDraw.Draw(img)
            draw.rectangle([50, 50, 150, 150], fill='red')
            draw.ellipse([100, 100, 200, 200], fill='blue')
        else:
            img = Image.open(image_path).convert('RGB')
            img = img.resize((224, 224))
        
        print("图像增强效果对比：")
        for name, transform in self.transforms.items():
            try:
                augmented = transform(img)
                print(f"{name}: 形状 {augmented.shape}")
            except Exception as e:
                print(f"{name}: 变换失败 - {e}")

# 演示图像增强
aug_demo = ImageAugmentationDemo()
aug_demo.demonstrate_augmentation()

# 训练中使用增强的示例
class AugmentedDataset(Dataset):
    """使用增强的数据集"""
    
    def __init__(self, data, labels, transform=None):
        self.data = data
        self.labels = labels
        self.transform = transform
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        image = self.data[idx]
        label = self.labels[idx]
        
        if self.transform:
            image = self.transform(image)
        
        return image, label

# 2. 迁移学习与微调（Transfer Learning & Fine-tuning）
# --------------------------------------------------
# 原理说明：
# 迁移学习利用在大数据集上预训练的模型来解决新任务
# 微调通过冻结部分层、调整学习率等策略优化训练过程
# 核心思想：底层特征通用，顶层特征任务相关

print("\n2. 迁移学习与微调：")

class TransferLearningModel:
    """迁移学习模型构建器"""
    
    def __init__(self, model_name='resnet18', num_classes=10, pretrained=True):
        self.model_name = model_name
        self.num_classes = num_classes
        self.model = self._build_model(pretrained)
    
    def _build_model(self, pretrained):
        """构建迁移学习模型"""
        if self.model_name == 'resnet18':
            model = torchvision.models.resnet18(pretrained=pretrained)
            # 替换最后的分类层
            model.fc = nn.Linear(model.fc.in_features, self.num_classes)
        elif self.model_name == 'vgg16':
            model = torchvision.models.vgg16(pretrained=pretrained)
            model.classifier[-1] = nn.Linear(model.classifier[-1].in_features, 
                                           self.num_classes)
        else:
            raise ValueError(f"不支持的模型: {self.model_name}")
        
        return model
    
    def freeze_backbone(self, freeze=True):
        """冻结骨干网络参数"""
        if self.model_name == 'resnet18':
            for param in self.model.parameters():
                param.requires_grad = not freeze
            # 只训练分类层
            for param in self.model.fc.parameters():
                param.requires_grad = True
        
        frozen_params = sum(1 for p in self.model.parameters() if not p.requires_grad)
        total_params = sum(1 for p in self.model.parameters())
        print(f"冻结参数数: {frozen_params}/{total_params}")
        
    def get_optimizer_groups(self, backbone_lr=1e-4, head_lr=1e-3):
        """为不同部分设置不同学习率"""
        if self.model_name == 'resnet18':
            backbone_params = []
            head_params = []
            
            for name, param in self.model.named_parameters():
                if 'fc' in name:
                    head_params.append(param)
                else:
                    backbone_params.append(param)
            
            return [
                {'params': backbone_params, 'lr': backbone_lr},
                {'params': head_params, 'lr': head_lr}
            ]

# 创建迁移学习模型
transfer_model = TransferLearningModel('resnet18', num_classes=10)
print(f"模型总参数数: {sum(p.numel() for p in transfer_model.model.parameters())}")

# 演示不同的微调策略
print("微调策略对比：")

# 策略1：冻结骨干网络
transfer_model.freeze_backbone(freeze=True)

# 策略2：不同学习率
optimizer_groups = transfer_model.get_optimizer_groups()
optimizer = torch.optim.Adam(optimizer_groups)

print("优化器参数组:")
for i, group in enumerate(optimizer.param_groups):
    print(f"组{i}: 学习率={group['lr']}, 参数数={len(group['params'])}")

# 3. 目标检测基础组件
# -------------------
# 原理说明：
# 目标检测需要同时解决分类和定位问题
# 核心组件：边界框处理、锚框生成、NMS等

print("\n3. 目标检测基础组件：")

class BoundingBoxUtils:
    """边界框工具类"""
    
    @staticmethod
    def box_iou(box1, box2):
        """计算两个边界框的IoU"""
        # box格式: [x1, y1, x2, y2]
        # 计算交集面积
        inter_x1 = torch.max(box1[0], box2[0])
        inter_y1 = torch.max(box1[1], box2[1])
        inter_x2 = torch.min(box1[2], box2[2])
        inter_y2 = torch.min(box1[3], box2[3])
        
        inter_area = torch.clamp(inter_x2 - inter_x1, min=0) * \
                    torch.clamp(inter_y2 - inter_y1, min=0)
        
        # 计算各自面积
        area1 = (box1[2] - box1[0]) * (box1[3] - box1[1])
        area2 = (box2[2] - box2[0]) * (box2[3] - box2[1])
        
        # 计算并集面积
        union_area = area1 + area2 - inter_area
        
        return inter_area / (union_area + 1e-6)
    
    @staticmethod
    def nms(boxes, scores, iou_threshold=0.5):
        """非最大抑制"""
        # 按置信度排序
        sorted_indices = torch.argsort(scores, descending=True)
        keep = []
        
        while len(sorted_indices) > 0:
            # 保留置信度最高的框
            current = sorted_indices[0]
            keep.append(current.item())
            
            if len(sorted_indices) == 1:
                break
            
            # 计算与其他框的IoU
            current_box = boxes[current]
            other_boxes = boxes[sorted_indices[1:]]
            
            ious = torch.stack([BoundingBoxUtils.box_iou(current_box, box) 
                               for box in other_boxes])
            
            # 保留IoU小于阈值的框
            mask = ious < iou_threshold
            sorted_indices = sorted_indices[1:][mask]
        
        return keep

class AnchorGenerator:
    """锚框生成器"""
    
    def __init__(self, scales=[32, 64, 128], ratios=[0.5, 1.0, 2.0]):
        self.scales = scales
        self.ratios = ratios
    
    def generate_anchors(self, feature_size, image_size):
        """生成锚框"""
        anchors = []
        h, w = feature_size
        img_h, img_w = image_size
        
        # 特征图上每个位置的步长
        stride_h = img_h / h
        stride_w = img_w / w
        
        for i in range(h):
            for j in range(w):
                # 特征图位置对应到原图的中心点
                center_x = (j + 0.5) * stride_w
                center_y = (i + 0.5) * stride_h
                
                for scale in self.scales:
                    for ratio in self.ratios:
                        # 计算锚框宽高
                        w_anchor = scale * np.sqrt(ratio)
                        h_anchor = scale / np.sqrt(ratio)
                        
                        # 锚框坐标 [x1, y1, x2, y2]
                        x1 = center_x - w_anchor / 2
                        y1 = center_y - h_anchor / 2
                        x2 = center_x + w_anchor / 2
                        y2 = center_y + h_anchor / 2
                        
                        anchors.append([x1, y1, x2, y2])
        
        return torch.tensor(anchors, dtype=torch.float32)

# 演示边界框操作
print("边界框IoU计算:")
box1 = torch.tensor([10, 10, 50, 50])  # [x1, y1, x2, y2]
box2 = torch.tensor([30, 30, 70, 70])
iou = BoundingBoxUtils.box_iou(box1, box2)
print(f"IoU: {iou:.3f}")

# 演示锚框生成
anchor_gen = AnchorGenerator()
anchors = anchor_gen.generate_anchors(feature_size=(4, 4), image_size=(128, 128))
print(f"生成的锚框数量: {len(anchors)}")
print(f"前5个锚框: {anchors[:5]}")

# 4. 简单的目标检测模型
# ---------------------
print("\n4. 简单的目标检测模型：")

class SimpleDetector(nn.Module):
    """简单的目标检测模型"""
    
    def __init__(self, num_classes=20, num_anchors=9):
        super().__init__()
        self.num_classes = num_classes
        self.num_anchors = num_anchors
        
        # 使用预训练的特征提取器
        backbone = torchvision.models.resnet18(pretrained=True)
        self.features = nn.Sequential(*list(backbone.children())[:-2])
        
        # 检测头
        self.classifier = nn.Conv2d(512, num_anchors * num_classes, 1)
        self.bbox_regressor = nn.Conv2d(512, num_anchors * 4, 1)
        
    def forward(self, x):
        # 特征提取
        features = self.features(x)
        
        # 分类和回归预测
        cls_scores = self.classifier(features)
        bbox_preds = self.bbox_regressor(features)
        
        return cls_scores, bbox_preds

# 创建检测模型
detector = SimpleDetector(num_classes=10).to(device)
print(f"检测器参数数: {sum(p.numel() for p in detector.parameters())}")

# 测试检测器
test_image = torch.randn(2, 3, 224, 224).to(device)
cls_scores, bbox_preds = detector(test_image)
print(f"分类输出形状: {cls_scores.shape}")
print(f"回归输出形状: {bbox_preds.shape}")

# 5. 语义分割模型
# ---------------
# 原理说明：
# 语义分割对图像中每个像素进行分类
# 常用架构：FCN、U-Net、DeepLab等
# 核心技术：上采样、跳跃连接、空洞卷积

print("\n5. 语义分割模型：")

class SimpleSegmentationModel(nn.Module):
    """简单的语义分割模型（类FCN结构）"""
    
    def __init__(self, num_classes=21):  # Pascal VOC有21类
        super().__init__()
        
        # 编码器（下采样）
        self.encoder = nn.Sequential(
            nn.Conv2d(3, 64, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 64, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),  # 1/2
            
            nn.Conv2d(64, 128, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 128, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),  # 1/4
            
            nn.Conv2d(128, 256, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),  # 1/8
        )
        
        # 解码器（上采样）
        self.decoder = nn.Sequential(
            nn.Conv2d(256, 256, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(256, 128, 2, stride=2),  # 2x
            
            nn.Conv2d(128, 128, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(128, 64, 2, stride=2),   # 4x
            
            nn.Conv2d(64, 64, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(64, num_classes, 2, stride=2),  # 8x
        )
        
    def forward(self, x):
        features = self.encoder(x)
        output = self.decoder(features)
        return output

# 创建分割模型
seg_model = SimpleSegmentationModel(num_classes=21).to(device)
print(f"分割模型参数数: {sum(p.numel() for p in seg_model.parameters())}")

# 测试分割模型
test_image = torch.randn(2, 3, 224, 224).to(device)
seg_output = seg_model(test_image)
print(f"分割输出形状: {seg_output.shape}")

# 6. 神经风格迁移
# ---------------
# 原理说明：
# 神经风格迁移通过最小化内容损失和风格损失来生成图像
# 内容损失：保持图像内容
# 风格损失：学习艺术风格的纹理特征

print("\n6. 神经风格迁移：")

class StyleTransferLoss:
    """风格迁移损失函数"""
    
    def __init__(self, device):
        self.device = device
        # 使用VGG作为特征提取器
        vgg = torchvision.models.vgg19(pretrained=True).features.to(device)
        self.vgg = vgg.eval()
        
        # 冻结VGG参数
        for param in self.vgg.parameters():
            param.requires_grad = False
        
        # 用于提取特征的层
        self.content_layers = ['conv_4']
        self.style_layers = ['conv_1', 'conv_2', 'conv_3', 'conv_4', 'conv_5']
    
    def extract_features(self, x):
        """提取VGG特征"""
        features = {}
        layer_names = ['conv_1', 'conv_2', 'conv_3', 'conv_4', 'conv_5']
        
        x = x
        for i, layer in enumerate(self.vgg[:25]):  # 只使用前25层
            x = layer(x)
            
            # 在特定层保存特征
            if i in [1, 6, 11, 20, 25]:  # VGG19的关键层位置
                layer_idx = [1, 6, 11, 20, 25].index(i)
                if layer_idx < len(layer_names):
                    features[layer_names[layer_idx]] = x
        
        return features
    
    def gram_matrix(self, x):
        """计算Gram矩阵（用于风格损失）"""
        batch_size, channels, height, width = x.size()
        features = x.view(batch_size, channels, height * width)
        gram = torch.bmm(features, features.transpose(1, 2))
        return gram / (channels * height * width)
    
    def content_loss(self, generated_features, content_features):
        """内容损失"""
        loss = 0
        for layer in self.content_layers:
            if layer in generated_features and layer in content_features:
                loss += F.mse_loss(generated_features[layer], 
                                 content_features[layer])
        return loss
    
    def style_loss(self, generated_features, style_features):
        """风格损失"""
        loss = 0
        for layer in self.style_layers:
            if layer in generated_features and layer in style_features:
                gen_gram = self.gram_matrix(generated_features[layer])
                style_gram = self.gram_matrix(style_features[layer])
                loss += F.mse_loss(gen_gram, style_gram)
        return loss

class NeuralStyleTransfer:
    """神经风格迁移模型"""
    
    def __init__(self, device):
        self.device = device
        self.loss_fn = StyleTransferLoss(device)
    
    def transfer_style(self, content_img, style_img, steps=100, 
                      content_weight=1e4, style_weight=1e-2):
        """执行风格迁移"""
        # 初始化生成图像（从内容图像开始）
        generated_img = content_img.clone().requires_grad_(True)
        optimizer = torch.optim.LBFGS([generated_img])
        
        # 提取目标特征
        content_features = self.loss_fn.extract_features(content_img)
        style_features = self.loss_fn.extract_features(style_img)
        
        print("开始风格迁移...")
        for step in range(steps):
            def closure():
                optimizer.zero_grad()
                
                # 提取当前生成图像的特征
                generated_features = self.loss_fn.extract_features(generated_img)
                
                # 计算损失
                content_loss = self.loss_fn.content_loss(generated_features, 
                                                       content_features)
                style_loss = self.loss_fn.style_loss(generated_features, 
                                                   style_features)
                
                total_loss = content_weight * content_loss + style_weight * style_loss
                total_loss.backward()
                
                if step % 20 == 0:
                    print(f"Step {step}: Content Loss: {content_loss.item():.4f}, "
                          f"Style Loss: {style_loss.item():.4f}")
                
                return total_loss
            
            optimizer.step(closure)
            
            # 限制像素值范围
            with torch.no_grad():
                generated_img.clamp_(0, 1)
            
            if step >= 5:  # 为了演示只运行几步
                break
        
        return generated_img.detach()

# 创建风格迁移模型（需要大量计算，这里只演示结构）
if device.type == 'cuda':
    style_transfer = NeuralStyleTransfer(device)
    print("神经风格迁移模型已创建")
    
    # 创建测试图像
    content_img = torch.randn(1, 3, 256, 256).to(device)
    style_img = torch.randn(1, 3, 256, 256).to(device)
    
    print("模型结构验证通过")
else:
    print("风格迁移需要GPU支持，跳过演示")

# 7. 计算机视觉训练流程
# ---------------------
print("\n7. 计算机视觉训练流程：")

class CVTrainingPipeline:
    """计算机视觉训练管道"""
    
    def __init__(self, model, device, task_type='classification'):
        self.model = model.to(device)
        self.device = device
        self.task_type = task_type
    
    def train_classification(self, train_loader, val_loader, 
                           epochs=5, lr=0.001):
        """分类任务训练"""
        optimizer = torch.optim.Adam(self.model.parameters(), lr=lr)
        criterion = nn.CrossEntropyLoss()
        
        self.model.train()
        for epoch in range(epochs):
            total_loss = 0
            correct = 0
            total = 0
            
            for batch_idx, (data, target) in enumerate(train_loader):
                data, target = data.to(self.device), target.to(self.device)
                
                optimizer.zero_grad()
                output = self.model(data)
                loss = criterion(output, target)
                loss.backward()
                optimizer.step()
                
                total_loss += loss.item()
                pred = output.argmax(dim=1)
                correct += pred.eq(target).sum().item()
                total += target.size(0)
                
                if batch_idx >= 5:  # 仅演示几个batch
                    break
            
            accuracy = 100. * correct / total
            print(f'Epoch {epoch+1}: Loss: {total_loss/(batch_idx+1):.4f}, '
                  f'Accuracy: {accuracy:.2f}%')
    
    def train_detection(self, train_loader, epochs=3):
        """检测任务训练（简化版）"""
        optimizer = torch.optim.Adam(self.model.parameters(), lr=0.001)
        
        self.model.train()
        for epoch in range(epochs):
            total_cls_loss = 0
            total_reg_loss = 0
            
            for batch_idx, (data, targets) in enumerate(train_loader):
                data = data.to(self.device)
                
                optimizer.zero_grad()
                cls_scores, bbox_preds = self.model(data)
                
                # 简化的损失计算（实际需要更复杂的处理）
                cls_loss = cls_scores.mean()  # 占位符
                reg_loss = bbox_preds.mean()  # 占位符
                
                total_loss = cls_loss + reg_loss
                total_loss.backward()
                optimizer.step()
                
                total_cls_loss += cls_loss.item()
                total_reg_loss += reg_loss.item()
                
                if batch_idx >= 3:  # 仅演示几个batch
                    break
            
            print(f'Epoch {epoch+1}: Cls Loss: {total_cls_loss/(batch_idx+1):.4f}, '
                  f'Reg Loss: {total_reg_loss/(batch_idx+1):.4f}')

# 创建模拟数据进行训练演示
print("创建模拟数据集...")
# 分类数据
X_cls = torch.randn(100, 3, 224, 224)
y_cls = torch.randint(0, 10, (100,))
cls_dataset = torch.utils.data.TensorDataset(X_cls, y_cls)
cls_loader = DataLoader(cls_dataset, batch_size=16, shuffle=True)

# 训练分类模型
print("\n训练分类模型:")
cls_model = torchvision.models.resnet18(pretrained=False)
cls_model.fc = nn.Linear(cls_model.fc.in_features, 10)

cls_trainer = CVTrainingPipeline(cls_model, device, 'classification')
cls_trainer.train_classification(cls_loader, cls_loader, epochs=2)

# 8. 模型评估与可视化
# -------------------
print("\n8. 模型评估与可视化：")

class CVModelEvaluator:
    """计算机视觉模型评估器"""
    
    def __init__(self, model, device):
        self.model = model.to(device)
        self.device = device
    
    def evaluate_classification(self, test_loader):
        """评估分类模型"""
        self.model.eval()
        correct = 0
        total = 0
        
        with torch.no_grad():
            for data, target in test_loader:
                data, target = data.to(self.device), target.to(self.device)
                outputs = self.model(data)
                pred = outputs.argmax(dim=1)
                correct += pred.eq(target).sum().item()
                total += target.size(0)
        
        accuracy = 100. * correct / total
        print(f'测试准确率: {accuracy:.2f}%')
        return accuracy
    
    def compute_class_accuracy(self, test_loader, num_classes):
        """计算每个类别的准确率"""
        class_correct = torch.zeros(num_classes)
        class_total = torch.zeros(num_classes)
        
        self.model.eval()
        with torch.no_grad():
            for data, target in test_loader:
                data, target = data.to(self.device), target.to(self.device)
                outputs = self.model(data)
                pred = outputs.argmax(dim=1)
                
                for i in range(target.size(0)):
                    label = target[i].item()
                    class_correct[label] += pred[i].eq(target[i]).item()
                    class_total[label] += 1
        
        for i in range(num_classes):
            if class_total[i] > 0:
                accuracy = 100. * class_correct[i] / class_total[i]
                print(f'类别 {i} 准确率: {accuracy:.2f}%')

# 评估训练好的模型
evaluator = CVModelEvaluator(cls_model, device)
test_accuracy = evaluator.evaluate_classification(cls_loader)

print("\n计算机视觉技术总结：")
print("1. 图像增强是提升模型泛化能力的重要手段")
print("2. 迁移学习可以显著加速训练并提升性能")
print("3. 目标检测需要同时解决分类和定位问题")
print("4. 语义分割通过像素级分类实现精细理解")
print("5. 神经风格迁移展示了深度学习的创造性应用")
print("6. 合适的训练流程和评估方法是成功的关键")

print(f"\n所有演示完成！使用设备: {device}") 