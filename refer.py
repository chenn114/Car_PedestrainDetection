import os
from ultralytics import YOLO


def run_inference():
    # 1. 设置权重文件路径
    # 使用 r 前缀防止 Windows 路径转义，指向你之前设置的 D 盘目录
    model_path = r'D:\AIData\yolo11_vp\weights\best.pt'

    # 安全检查：确保模型文件确实存在
    if not os.path.exists(model_path):
        print(f"❌ 错误：在 {model_path} 未找到权重文件，请检查训练是否完成。")
        return

    # 2. 加载训练好的模型
    model = YOLO(model_path)

    # 3. 对图片进行推理
    # source 可以是单张图片、文件夹、甚至是视频/摄像头
    source_image = 'test.jpg'

    print(f"🔍 正在处理: {source_image} ...")
    results = model.predict(
        source=source_image,
        conf=0.25,  # 置信度阈值：只显示概率大于 25% 的目标
        iou=0.45,  # NMS(非极大值抑制)阈值：处理重叠框，防止一个目标出两个框
        save=True,  # 自动保存结果图到 runs/detect/predict 文件夹
        device='cpu',  # 强制使用 CPU
        line_width=2  # 绘图线宽，方便实验报告截图
    )

    # 4. 解析并打印检测到的目标信息
    print("\n" + "=" * 30)
    print("📊 检测结果摘要：")
    for r in results:
        # 获取当前图片的检测框数量
        num_boxes = len(r.boxes)
        print(f"图片 '{source_image}' 中检测到 {num_boxes} 个目标。")

        for box in r.boxes:
            cls_id = int(box.cls[0])  # 类别索引
            conf_val = float(box.conf[0])  # 置信度数值
            name = model.names[cls_id]  # 类别名称（如 person, car 等）

            # 获取坐标 (xyxy 格式)
            coords = box.xyxy[0].tolist()

            print(f"- 【{name}】 置信度: {conf_val:.2f} | 位置: {[round(x, 1) for x in coords]}")

    print("=" * 30)
    print(f"✅ 结果图已保存至: {results[0].save_dir}")


if __name__ == "__main__":
    run_inference()