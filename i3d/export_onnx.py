import torch
import torch.nn as nn
from pytorch_i3d import InceptionI3d
import os

class I3DWrapper(nn.Module):
    """
    包装类：确保 ONNX 输出的结果与 extract_features.py 中的处理逻辑一致
    """
    def __init__(self, model, extract_features=True):
        super(I3DWrapper, self).__init__()
        self.model = model
        self.extract_features_mode = extract_features

    def forward(self, x):
        if self.extract_features_mode:
            # 模拟 extract_features.py 中的逻辑
            features = self.model.extract_features(x)
            # 对 Dim 2, 3, 4 (Time, Height, Width) 求均值
            out = torch.mean(features, dim=(2, 3, 4))
            return out
        else:
            # 返回原始 Logits
            return self.model(x)

def export_onnx():
    # 1. 参数设置 (根据 extract_features.py)
    num_classes = 26
    dropout_rate = 0.7
    checkpoint_path = r'models/a320.pt'
    output_path = r'models/a320.onnx'
    
    if not os.path.exists(checkpoint_path):
        print(f"错误: 找不到模型文件 {checkpoint_path}")
        # 如果找不到 a320.pt，尝试寻找 rgb_imagenet.pt 作为演示
        if os.path.exists('models/rgb_imagenet.pt'):
             checkpoint_path = 'models/rgb_imagenet.pt'
             print(f"使用 {checkpoint_path} 进行演示导出...")
        else:
             return

    # 2. 初始化模型架构
    model = InceptionI3d(
        num_classes=400, # 初始通常为 400 (Kinetics)
        in_channels=3,
        dropout_keep_prob=1 - dropout_rate
    )
    model.replace_logits(num_classes)

    # 3. 加载权重
    print(f"正在加载 {checkpoint_path}...")
    checkpoint = torch.load(checkpoint_path, map_location='cpu')
    
    # 根据 extract_features.py 中的逻辑处理 state_dict
    if isinstance(checkpoint, dict) and 'state_dict' in checkpoint:
        state_dict = checkpoint['state_dict']
        # 移除 'module.' 前缀 (用于 DataParallel 加载)
        new_state_dict = { (k[7:] if k.startswith('module.') else k): v for k, v in state_dict.items() }
        model.load_state_dict(new_state_dict)
    else:
        model.load_state_dict(checkpoint)

    model.eval()

    # 4. 包装模型以匹配推理输出
    # 如果您需要分类结果，设置 extract_features=False
    wrapped_model = I3DWrapper(model, extract_features=True)
    wrapped_model.eval()

    # 5. 定义 dummy input (Batch, Channel, Time, Height, Width)
    # 常规 I3D 输入为 16 或 64 帧，224x224 分辨率
    dummy_input = torch.randn(1, 3, 16, 224, 224)

    # 6. 导出
    print("正在导出为 ONNX...")
    torch.onnx.export(
        wrapped_model,
        dummy_input,
        output_path,
        export_params=True,
        opset_version=14, # 建议使用 12 以上版本以支持 3D 算子
        do_constant_folding=True,
        input_names=['input'],
        output_names=['output'],
        dynamic_axes={
            'input': {0: 'batch_size', 2: 'num_frames'}, # 允许 batch 和帧数动态
            'output': {0: 'batch_size'}
        }
    )
    print(f"成功导出模型至: {output_path}")

if __name__ == '__main__':
    export_onnx()
