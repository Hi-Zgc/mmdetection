import cv2
import onnxruntime as rt

# 读取 ONNX 模型
model_path = "model.onnx"
session = rt.InferenceSession(model_path)

# 打开摄像头
camera = cv2.VideoCapture(0)

# 循环读取摄像头帧
while True:
    ret, frame = camera.read()
    if not ret:
        break

    # 使用 ONNX Runtime 进行推理加速
    input_name = session.get_inputs()[0].name
    input_data = cv2.resize(frame, (224, 224))  # 将帧的大小调整为模型的输入大小
    input_data = input_data.transpose((2, 0, 1))  # 调整输入的维度顺序
    input_data = input_data.reshape((1, 3, 224, 224))  # 增加批量维度
    output_data = session.run(None, {input_name: input_data})[0]
    print(output_data)  # 输出推理结果

    # 显示摄像头帧
    cv2.imshow("Camera", frame)
    if cv2.waitKey(1) == 27:  # 按下 Esc 退出
        break

# 关闭摄像头
camera.release()
cv2.destroyAllWindows()