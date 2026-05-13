import logging
import os
import sys

# os.chdir("/home/swin/zyli/BackdoorAttack/WeatherApproximationGAN/shapley-adv")

# 创建记录器
logger = logging.getLogger('SmoothTrigger')
logger.setLevel(logging.DEBUG)

# 创建文件处理程序和控制台处理程序
os.makedirs("log/", exist_ok=True)
file_handler = logging.FileHandler(f'log/{os.path.splitext(os.path.basename(sys.argv[0]))[0]}.log', encoding="UTF-8")
console_handler = logging.StreamHandler()

# 定义格式化程序并添加到处理程序
formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
file_handler.setFormatter(formatter)
console_handler.setFormatter(formatter)

# 添加处理程序到记录器
logger.addHandler(file_handler)
logger.addHandler(console_handler)
