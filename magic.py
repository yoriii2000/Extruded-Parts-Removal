import numpy as np
import matplotlib.pyplot as plt

k0 = 0
k1 = 1
k2 = k1 / 2
tau0 = 0
Theta = np.linspace(0, 2*np.pi, 1000)
tau1 = tau0 + k1*np.sin(2*Theta)**2 + k2/4*np.sin(2*Theta)**4

k3 = 0
k4 = 1
k5 = -k4 / 2
tau2 = 0
Theta1 = np.linspace(0, 2*np.pi, 1000)
tau3 = tau2 + k4*np.sin(2*Theta1)**2 + k5/4*np.sin(2*Theta1)**4

plt.figure(figsize = (6, 4.5), dpi = 100)                 # 設定圖片尺寸
plt.xlabel('Theta', fontsize = 16)                        # 設定坐標軸標籤
plt.xticks(fontsize = 12)                                 # 設定坐標軸數字格式
plt.yticks(fontsize = 12)
plt.grid(color = 'red', linestyle = '--', linewidth = 1)  # 設定格線顏色、種類、寬度
plt.ylim(0, 1.2)                                          # 設定y軸繪圖範圍
# 繪圖並設定線條顏色、寬度、圖例
line1, = plt.plot(Theta, tau1, color = 'red', linewidth = 3, label = 'A')
line2, = plt.plot(Theta1, tau3, color = 'blue', linewidth = 3, label = 'B')
plt.legend(handles = [line1, line2], loc='upper right')
plt.show()