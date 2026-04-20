import os
from ultralytics import YOLO


def train_model():
    # 1. 初始化模型
    # 加载 YOLOv11 nano 预训练权重。n系列最轻量，非常适合 CPU 环境或初步实验。
    model = YOLO('yolo11n.pt')

    # 2. 获取数据集配置文件路径
    # 使用绝对路径，相对路径需要确保 train.py 与数据集文件夹在同一目录下
    data_path = 'D:\AIData\YOLO11\data.yaml'

    # 安全检查：防止路径错误导致程序直接崩溃
    if not os.path.exists(data_path):
        print(f"❌ 错误：找不到数据集配置文件 '{data_path}'，请检查文件夹名称是否正确。")
        return

    # 3. 开始模型微调 (Fine-tuning)
    print("🚀 开始训练...")
    results = model.train(
        data=data_path,  # 标注信息与路径配置
        epochs=10,  # 训练轮数。对于实验课，10轮足以观察到收敛趋势
        imgsz=416,  # 输入图像尺寸。416比默认的640更小，能显著提升CPU训练速度
        batch=4,  # 批大小。CPU训练建议设小（4-8），防止内存溢出
        device='cpu',  # 指定计算设备。若有显卡可改为 0 或 'cuda'
        workers=2,  # 数据加载的线程数。通常设为 CPU 核心数的一半
        project='runs/detect',  # 结果保存的总目录
        name='yolo11_vp',  # 本次实验的子目录名称
        val=True,  # 训练结束时自动运行验证集，方便填实验报告
        exist_ok=True  # 如果文件夹已存在，直接覆盖而不报错
    )

    print("-" * 30)
    print(f"✅ 训练完成！")
    print(f"📈 最佳权重路径: {results.save_dir}/weights/best.pt")
    print(f"📊 训练指标记录: {results.save_dir}/results.csv")


if __name__ == "__main__":
    train_model()