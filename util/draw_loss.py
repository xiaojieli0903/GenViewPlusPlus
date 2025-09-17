import re
import matplotlib.pyplot as plt

def extract_loss(file_path):
    losses = []
    with open(file_path, 'r') as f:
        for line in f:
            # 匹配 "loss: 数值" 的部分
            match = re.search(r'loss:\s*([0-9.]+)', line)
            if match:
                loss = float(match.group(1))
                losses.append(loss)
    return losses

# 修改为你的文件路径
file1 = '/ssd/lixiaojie/code/genview-clip/StableRep/output/100w-50wSyn/real-txt-img-imgtxt-Ada_QD_25e_gamma1/detailed_log.txt'
file2 = '/ssd/lixiaojie/code/genview-clip/StableRep/output/100w-50wSyn/real-txt-img-imgtxt-Ada_25e/detailed_log.txt'

losses1 = extract_loss(file1)
losses2 = extract_loss(file2)

# 画图
plt.figure(figsize=(10, 5))
plt.plot(losses1, label='Log 1', color='blue')
plt.plot(losses2, label='Log 2', color='orange')
plt.xlabel('Step')
plt.ylabel('Loss')
plt.title('Loss Comparison Between Two Logs')
plt.legend()
plt.grid(True)
plt.tight_layout()

save_path = '/ssd/lixiaojie/code/genview-clip/StableRep/loss_comparison.png'  # 修改为你想保存的路径
plt.savefig(save_path)