import os
import sys

print("===== 测试开始 =====")
print(f"当前工作目录: {os.getcwd()}")
print(f"Python版本: {sys.version}")

# 检查输入目录
input_dir = "D:/my_video_project/input"
if os.path.exists(input_dir):
    print(f"输入目录存在: {input_dir}")
    files = os.listdir(input_dir)
    print(f"目录中有 {len(files)} 个项目")
    for i, file in enumerate(files[:5]):  # 只显示前5个
        print(f"  {i+1}. {file}")
else:
    print(f"输入目录不存在: {input_dir}")

# 写入测试文件
try:
    with open("test_output.txt", "w") as f:
        f.write("这是一个测试\n")
    print("成功创建test_output.txt文件")
except Exception as e:
    print(f"创建文件失败: {str(e)}")

print("===== 测试结束 =====")
input("按回车键退出...")
