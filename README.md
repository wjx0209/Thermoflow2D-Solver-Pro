# Thermoflow2D-Solver-Pro
Manual for 2D Incompressible Flow with Heat Transfer Coupling
# 🌬️ Thermoflow2D Solver Pro

A 2D laminar fluid flow and heat transfer simulation tool with a modern graphical user interface (GUI), built using `Tkinter` and `ttkbootstrap`. This solver uses the vorticity-streamfunction formulation of the incompressible Navier-Stokes equations, coupled with the energy equation.

---

## 📁 Folder Structure
Thermoflow2D-Solver-Pro/
├── GUI.py # Main Python GUI application
└── User Manual.pdf # User guide with instructions and screenshot

---

## 💡 Features

- ✅ Interactive GUI with real-time simulation controls
- 🎨 Light/Dark theme toggle via `ttkbootstrap`
- 💾 Save and load simulation configurations (`.json`)
- 📈 Built-in animation of velocity fields and temperature distribution
- 📊 Line plots of horizontal/vertical velocities and temperature profiles
- 🧮 Solves:
  - 2D incompressible Navier-Stokes (vorticity-streamfunction)
  - 2D energy equation (heat transfer)

---

## 🔧 Numerical Methods

| Component           | Method                |
|---------------------|------------------------|
| Time Integration    | Explicit Euler         |
| Spatial Derivatives | Second-order Central Difference |
| Poisson Solver      | Successive Over-Relaxation (SOR) |
| Advection-Diffusion | Explicit scheme        |

---

## ▶️ How to Run

### 🐍 Requirements

- Python ≥ 3.8
- Packages:
  - `numpy`
  - `matplotlib`
  - `ttkbootstrap`

### 💻 Run the GUI

```bash
python GUI.py
您说：
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm, animation
from IPython.display import HTML

# --- 1. 设置流动参数（仅通道流） ---
Lx_input = 5.0     # 用户输入的长度 (可修改)
Ly_input = 1.0     # 用户输入的宽度 (可修改)
dx_target = 0.01   # 目标x方向网格间距
dy_target = 0.01   # 目标y方向网格间距
nx = int(np.ceil(Lx_input / dx_target)) + 1  # 网格数量x方向
ny = int(np.ceil(Ly_input / dy_target)) + 1  # 网格数量y方向
Lx = Lx_input      # 实际长度
Ly = Ly_input      # 实际宽度
Re = 100           # 雷诺数
dt, nt = 0.0005, 1000  # 时间步长和总步数

# 速度边界条件（通道流）
U_left = 1.0       # 入口速度
U_right = 0.0      # 出口速度（仅Dirichlet时使用）
U_top = 0.0        # 顶壁速度
U_bottom = 0.0     # 底壁速度
V_left = 0.0       # 入口y方向速度
V_right = 0.0      # 出口y方向速度
V_top = 0.0        # 顶壁y方向速度
V_bottom = 0.0     # 底壁y方向速度
velocity_bc_left = False   # 入口：固定速度（Dirichlet）
velocity_bc_right = True  # 出口：零梯度（Neumann）
velocity_bc_top = False    # 顶壁：固定速度（Dirichlet）
velocity_bc_bottom = False # 底壁：固定速度（Dirichlet）

# 选择非零速度计算 nu
U_default = 1.0  # 默认速度（当所有输入速度为零时使用）
velocities = [U_left, U_top, U_right, U_bottom, V_left, V_right, V_top, V_bottom]
non_zero_velocities = [v for v in velocities if v != 0]
if non_zero_velocities:
    U = non_zero_velocities[0]  # 取第一个非零速度
    print(f'使用速度 {U} 计算 nu')
else:
    U = U_default
    print(f'警告：所有输入速度为零，使用默认速度 {U_default} 计算 nu')

# 计算运动粘性系数
nu = U * Ly / Re
dx = Lx / (nx-1)   # x方向网格间距
dy = Ly / (ny-1)   # y方向网格间距

# --- 2. 设置热传递参数 ---
Pr = 0.71          # Prandtl数（空气）
alpha = nu / Pr    # 热扩散系数

# 温度边界条件
T_top = 300.0      # 顶壁温度 (K)
T_bottom = 30.0    # 底壁温度 (K)
T_left = 300.0     # 左壁温度 (K)
T_right = 270.0    # 右壁温度 (K)
adiabatic_top = False      # 顶壁固定温度（Dirichlet）
adiabatic_bottom = True    # 底壁绝热（Neumann）
adiabatic_left = True      # 左壁绝热（Neumann）
adiabatic_right = True     # 右壁绝热（Neumann）

# 检查数值稳定性
U_max = max(abs(v) for v in velocities)
if U_max == 0:
    U_max = U_default  # 使用默认速度确保稳定性
dt_adv = 0.5 * min(dx, dy) / U_max
dt_diff = 0.25 * min(dx*dx, dy*dy) / max(nu, alpha)
dt_safe = min(dt_adv, dt_diff)

# 输出模拟参数信息
print('Flow Type = channel')
print('Reynolds =', Re)
print('Prandtl =', Pr)
print('当前 dt =', dt)
print('安全 dt ≤', dt_safe)
print('流体扩散系数 =', nu)
print('热扩散系数 =', alpha)
print('网格数量 nx =', nx, ', ny =', ny)
print('实际网格间距 dx =', dx, ', dy =', dy)

# --- 3. 初始化场变量 ---
omega = np.zeros((ny, nx))      # 涡度场
psi = np.zeros((ny, nx))        # 流函数场
u_full = np.zeros((ny, nx))     # x方向速度场
v_full = np.zeros((ny, nx))     # y方向速度场
T = np.ones((ny, nx)) * T_bottom  # 初始温度场

# 历史记录列表
skip = nt // 200
psi_hist = []
omega_hist = []
u_hist = []
v_hist = []
T_hist = []

# 预计算SOR常数
coefdx = dx**2 / (2*(dx**2 + dy**2))
coefdy = dy**2 / (2*(dx**2 + dy**2))
coefdxdy = dx**2 * dy**2 / (2*(dx**2 + dy**2))

# --- 4. 边界条件函数 ---
def apply_velocity_bc(psi, omega, u_full, v_full):
    y = np.linspace(0, Ly, ny)
    psi[:, 0] = U_left * y  # 入口：Dirichlet，基于用户输入的 U_left
    psi[:, -1] = psi[:, -2]  # 出口：Neumann
    psi[0, :] = 0.0           # 底壁
    psi[-1, :] = U_left * Ly  # 顶壁，基于入口速度
    if velocity_bc_left:
        u_full[:, 0] = u_full[:, 1]
        v_full[:, 0] = v_full[:, 1]
    else:
        u_full[:, 0] = U_left
        v_full[:, 0] = V_left
    if velocity_bc_right:
        u_full[:, -1] = u_full[:, -2]
        v_full[:, -1] = v_full[:, -2]
    else:
        u_full[:, -1] = U_right
        v_full[:, -1] = V_right
    if velocity_bc_bottom:
        u_full[0, :] = u_full[1, :]
        v_full[0, :] = v_full[1, :]
    else:
        u_full[0, :] = U_bottom
        v_full[0, :] = V_bottom
    if velocity_bc_top:
        u_full[-1, :] = u_full[-2, :]
        v_full[-1, :] = v_full[-2, :]
    else:
        u_full[-1, :] = U_top
        v_full[-1, :] = V_top
    omega[:, 0] = 0.0
    omega[:, -1] = omega[:, -2]
    omega[0, :] = (psi[0, :] - psi[1, :]) * 2.0 / dy**2 - 2.0 * u_full[0, :] / dy
    omega[-1, :] = (psi[-1, :] - psi[-2, :]) * 2.0 / dy**2 - 2.0 * u_full[-1, :] / dy
    return psi, omega, u_full, v_full

def apply_temperature_bc(T):
    if adiabatic_top:
        T[-1, :] = T[-2, :]
    else:
        T[-1, :] = T_top
    if adiabatic_bottom:
        T[0, :] = T[1, :]
    else:
        T[0, :] = T_bottom
    if adiabatic_left:
        T[:, 0] = T[:, 1]
    else:
        T[:, 0] = T_left
    if adiabatic_right:
        T[:, -1] = T[:, -2]
    else:
        T[:, -1] = T_right
    return T

# --- 5. 主求解循环 ---
for p in range(nt):
    if p > 0:
        psi, omega, u_full, v_full = apply_velocity_bc(psi, omega, u_full, v_full)
        px = -(psi[2:,1:-1] - psi[:-2,1:-1]) * (omega[1:-1,2:] - omega[1:-1,:-2]) / (4*dy*dx)
        py = (omega[2:,1:-1] - omega[:-2,1:-1]) * (psi[1:-1,2:] - psi[1:-1,:-2]) / (4*dy*dx)
        pxy = (omega[1:-1,2:] + omega[1:-1,:-2] - 2*omega[1:-1,1:-1]) / dx**2 \
            + (omega[2:,1:-1] + omega[:-2,1:-1] - 2*omega[1:-1,1:-1]) / dy**2
        omega[1:-1,1:-1] += dt * (px + py + nu * pxy)
        err = 1.0
        counter = 0
        while err > 1e-5 and counter < 200:
            psi_old = psi.copy()
            psi[1:-1,1:-1] = coefdxdy * omega[1:-1,1:-1] \
                          + coefdx * (psi[2:,1:-1] + psi[:-2,1:-1]) \
                          + coefdy * (psi[1:-1,2:] + psi[1:-1,:-2])
            err = np.linalg.norm(psi - psi_old, ord=np.inf)
            counter += 1
        u_full[1:-1,1:-1] = (psi[2:,1:-1] - psi[:-2,1:-1]) / (2*dy)
        v_full[1:-1, 1:-1] = -(psi[1:-1, 2:] - psi[1:-1, :-2]) / (2*dx)
        psi, omega, u_full, v_full = apply_velocity_bc(psi, omega, u_full, v_full)
        T = apply_temperature_bc(T)
        px = -(psi[2:,1:-1] - psi[:-2,1:-1]) * (T[1:-1,2:] - T[1:-1,:-2]) / (4*dy*dx)
        py = (T[2:,1:-1] - T[:-2,1:-1]) * (psi[1:-1,2:] - psi[1:-1,:-2]) / (4*dy*dx)
        pxy = (T[1:-1,2:] + T[1:-1,:-2] - 2*T[1:-1,1:-1]) / dx**2 \
            + (T[2:,1:-1] + T[:-2,1:-1] - 2*T[1:-1,1:-1]) / dy**2
        T[1:-1,1:-1] += dt * (px + py + alpha * pxy)
    if p % skip == 0:
        psi_hist.append(psi.copy())
        omega_hist.append(omega.copy())
        u_hist.append(u_full.copy())
        v_hist.append(v_full.copy())
        T_hist.append(T.copy())

# --- 6. 创建动画 ---
x = np.linspace(0, Lx, nx)
y = np.linspace(0, Ly, ny)
X, Y = np.meshgrid(x, y)

# 中心线位置
y_slice = Ly / 2
j_slice = int(np.round(y_slice / Ly * (ny - 1)))

# 修改布局：增大画布尺寸并调整间距
fig, axes = plt.subplots(2, 3, figsize=(20, 12), dpi=100)
fig.patch.set_facecolor('white')
axes = axes.flatten()

plots = {
    'mag': {'ax': axes[0], 'contour': None, 'stream': None, 
            'title': 'Velocity Magnitude and Streamlines',
            'levels': np.arange(0, U_max + 0.1, 0.1),
            'cbar_label': 'Velocity Magnitude (m/s)'},
    
    'u': {'ax': axes[1], 'contour': None,
          'title': 'Horizontal Velocity (u)',
          'levels': np.arange(-U_max*0.5, U_max + 0.1, 0.1),
          'cbar_label': 'u-Velocity (m/s)'},
    
    'v': {'ax': axes[2], 'contour': None,
          'title': 'Vertical Velocity (v)',
          'levels': np.arange(-U_max*0.3, U_max*0.3 + 0.01, 0.03),
          'cbar_label': 'v-Velocity (m/s)'},
    
    'T': {'ax': axes[3], 'contour': None,
          'title': 'Temperature Field',
          'levels': np.linspace(min(T_bottom, T_left, T_right, T_top), 
                               max(T_bottom, T_left, T_right, T_top), 21),
          'cbar_label': 'Temperature (K)'},
    
    'line': {'ax': axes[4], 'lines': None,
             'title': f'y={y_slice:.2f} Velocity Distribution',
             'xlabel': 'x (m)', 'ylabel': 'Velocity (m/s)'},
    
    'T_line': {'ax': axes[5], 'lines': None,
               'title': f'y={y_slice:.2f} Temperature Distribution',
               'xlabel': 'x (m)', 'ylabel': 'Temperature (K)'}
}

for key in plots:
    ax = plots[key]['ax']
    ax.set_title(plots[key]['title'])
    if key not in ['line', 'T_line']:
        ax.set_aspect('equal')
        ax.set_xlabel('x (m)')
        ax.set_ylabel('y (m)')
        ax.set_xlim(0, Lx)
        ax.set_ylim(0, Ly)
        plots[key]['contour'] = ax.contourf(X, Y, np.zeros_like(X), 
                                            levels=plots[key]['levels'], 
                                            cmap=cm.jet, extend='both')
        fig.colorbar(plots[key]['contour'], ax=ax, 
                     orientation='vertical', 
                     label=plots[key]['cbar_label'])
    else:
        ax.set_xlabel(plots[key]['xlabel'])
        ax.set_ylabel(plots[key]['ylabel'])
        ax.grid(True)

def update(frame):
    u = u_hist[frame]
    v = v_hist[frame]
    T = T_hist[frame]
    mag = np.sqrt(u**2 + v**2)
    
    fig.suptitle(f'Time Step: {frame}/{len(u_hist)-1}', y=1.02, fontsize=12)
    
    if plots['mag']['contour'] is not None:
        for coll in plots['mag']['contour'].collections:
            coll.remove()
    plots['mag']['contour'] = plots['mag']['ax'].contourf(
        X, Y, mag, levels=plots['mag']['levels'], cmap=cm.jet, extend='both')
    
    if plots['mag']['stream'] is not None:
        plots['mag']['stream'].lines.remove()
    plots['mag']['stream'] = plots['mag']['ax'].streamplot(
        X, Y, u, v, density=1.0, color='w', linewidth=1.0)
    
    if plots['u']['contour'] is not None:
        for coll in plots['u']['contour'].collections:
            coll.remove()
    plots['u']['contour'] = plots['u']['ax'].contourf(
        X, Y, u, levels=plots['u']['levels'], cmap=cm.jet, extend='both')
    
    if plots['v']['contour'] is not None:
        for coll in plots['v']['contour'].collections:
            coll.remove()
    plots['v']['contour'] = plots['v']['ax'].contourf(
        X, Y, v, levels=plots['v']['levels'], cmap=cm.jet, extend='both')
    
    if plots['T']['contour'] is not None:
        for coll in plots['T']['contour'].collections:
            coll.remove()
    plots['T']['contour'] = plots['T']['ax'].contourf(
        X, Y, T, levels=plots['T']['levels'], cmap=cm.jet, extend='both')
    
    # 更新速度一维线图 (u, v)
    if plots['line']['lines'] is not None:
        for line in plots['line']['lines']:
            line.remove()
    plots['line']['lines'] = []
    plots['line']['lines'].append(plots['line']['ax'].plot(x, u[j_slice, :], 'r-', label='u')[0])
    plots['line']['lines'].append(plots['line']['ax'].plot(x, v[j_slice, :], 'g-', label='v')[0])
    plots['line']['ax'].legend()
    
    # 更新温度一维线图 (T)
    if plots['T_line']['lines'] is not None:
        for line in plots['T_line']['lines']:
            line.remove()
    plots['T_line']['lines'] = []
    plots['T_line']['lines'].append(plots['T_line']['ax'].plot(x, T[j_slice, :], 'k-', label='T')[0])
    plots['T_line']['ax'].legend()
    
    return [plots[key]['contour'] for key in plots if key not in ['line', 'T_line']] + \
           [plots['mag']['stream'].lines] + plots['line']['lines'] + plots['T_line']['lines']

# 创建动画
frame_step = max(1, len(u_hist) // 40)
anim = animation.FuncAnimation(
    fig, update,
    frames=range(0, len(u_hist), frame_step),
    init_func=lambda: update(0),
    interval=200,
    blit=False
)

# 调整子图间距
plt.tight_layout(pad=2.0, w_pad=2.0, h_pad=2.0)

plt.close(fig)
HTML(anim.to_jshtml())这个代码用了什么数值方法


ChatGPT 说：
你的这段代码是一个二维不可压缩Navier-Stokes + 能量方程的显式时间推进 + SOR迭代解Poisson方程的数值求解器。具体来说，它使用了以下数值方法：

🧮 数值方法综述：
1. 速度-涡度-流函数（Vorticity-Streamfunction）格式
避免直接解速度和压力，转而解涡度 ω 和流函数 ψ。

好处：自动满足不可压缩条件 
∇
⋅
𝑢
=
0
∇⋅u=0，不需要求解压力。

🔁 时间推进：显式Euler方法
对涡度传输方程和能量方程都使用了显式Euler差分法：

𝜔
𝑛
+
1
=
𝜔
𝑛
+
Δ
𝑡
⋅
𝑅
𝐻
𝑆
(
𝜔
𝑛
,
𝜓
𝑛
)
ω 
n+1
 =ω 
n
 +Δt⋅RHS(ω 
n
 ,ψ 
n
 )
𝑇
𝑛
+
1
=
𝑇
𝑛
+
Δ
𝑡
⋅
𝑅
𝐻
𝑆
(
𝑇
𝑛
,
𝜓
𝑛
)
T 
n+1
 =T 
n
 +Δt⋅RHS(T 
n
 ,ψ 
n
 )
📐 空间离散：二阶中心差分
对流项（如 u∂ω/∂x）和扩散项（如 ν∇²ω）都使用了中心差分：

一阶导数：二阶中心差分（用于对流项）

二阶导数：标准5点差分模板（用于扩散项）

💡 流函数方程（Poisson 方程）：SOR迭代
使用**松弛迭代法（Successive Over-Relaxation, SOR）**解：

∇
2
𝜓
=
−
𝜔
∇ 
2
 ψ=−ω
这部分你用了手动构造的SOR模板，使用固定迭代次数和误差容限控制收敛。

🔥 热传导方程：显式格式
能量守恒（温度）采用了如下形式：

∂
𝑇
∂
𝑡
+
𝑢
∂
𝑇
∂
𝑥
+
𝑣
∂
𝑇
∂
𝑦
=
𝛼
(
∂
2
𝑇
∂
𝑥
2
+
∂
2
𝑇
∂
𝑦
2
)
∂t
∂T
​
 +u 
∂x
∂T
​
 +v 
∂y
∂T
​
 =α( 
∂x 
2
 
∂ 
2
 T
​
 + 
∂y 
2
 
∂ 
2
 T
​
 )
显式欧拉法 + 中心差分。

✅ 边界条件处理
速度边界条件：Dirichlet 或 Neumann，通过手动赋值和边界公式处理。

温度边界条件：显式处理绝热边界（Neumann）与定温边界（Dirichlet）。

📌 总结表格：
物理方程	数值方法	时间格式	空间格式
涡度传输方程	显式Euler + 中心差分	显式	二阶中心差分
Poisson方程（求 ψ）	SOR迭代	伪稳态	二阶中心差分
能量方程	显式Euler	显式	二阶中心差分

如果你还想升级为更高精度或更稳定的方法（如ADI、Crank-Nicolson、投影法），我也可以帮你实现。是否需要介绍这些改进方向？







您说：

这个是什么数值方法。向前差分吗

ChatGPT 说：
你这张图展示的是：

✅ 显式Euler方法（Explicit Euler Method）
公式为：

𝜔
𝑛
+
1
=
𝜔
𝑛
+
Δ
𝑡
⋅
𝑅
𝐻
𝑆
(
𝜔
𝑛
,
𝜓
𝑛
)
ω 
n+1
 =ω 
n
 +Δt⋅RHS(ω 
n
 ,ψ 
n
 )
𝑇
𝑛
+
1
=
𝑇
𝑛
+
Δ
𝑡
⋅
𝑅
𝐻
𝑆
(
𝑇
𝑛
,
𝜓
𝑛
)
T 
n+1
 =T 
n
 +Δt⋅RHS(T 
n
 ,ψ 
n
 )
🧭 这是一种什么差分方法？
是的，它属于时间方向的向前差分（Forward Difference），也称：

前向欧拉法（Forward Euler）

属于一阶显式时间积分方法

对应时间导数的离散形式如下：

𝑇
𝑛
+
1
−
𝑇
𝑛
Δ
𝑡
=
𝑅
𝐻
𝑆
(
𝑇
𝑛
,
𝜓
𝑛
)
Δt
T 
n+1
 −T 
n
 
​
 =RHS(T 
n
 ,ψ 
n
 )
这正是向前差分（forward difference）的一种实现方式。

📌 特点总结：
项目	内容
方法名	显式Euler法（Explicit Euler）
类型	时间上显式、空间上通常用中心差分
时间差分格式	向前差分
优点	实现简单，计算代价低
缺点	条件稳定，需要满足CFL条件（受限时间步长）

如需将其替换为更稳定的**Crank-Nicolson（隐式）、Runge-Kutta（二阶或四阶）**方法，我也可以帮你改写。是否需要？







您说：

这个是什么数值方法

ChatGPT 说：
你这张图描述的是：

✅ 空间离散方法：二阶中心差分（Second-order Central Difference）
用于离散 PDE 中的空间导数项，图中总结如下：

📌 应用：
项目	使用方法	说明
对流项（如 
𝑢
∂
𝜔
∂
𝑥
u 
∂x
∂ω
​
 ）	一阶导数 → 二阶中心差分	对流导数使用中心差分提高精度（而非迎风差分）
扩散项（如 
𝜈
∇
2
𝑢
ν∇ 
2
 u）	二阶导数 → 5点差分模板	即标准拉普拉斯算子离散形式

✏️ 举例公式：
✅ 一阶导数（对流项）：
∂
𝑓
∂
𝑥
∣
𝑖
≈
𝑓
𝑖
+
1
−
𝑓
𝑖
−
1
2
Δ
𝑥
∂x
∂f
​
  
​
  
i
​
 ≈ 
2Δx
f 
i+1
​
 −f 
i−1
​
 
​
 
✅ 二阶导数（扩散项）：
∂
2
𝑓
∂
𝑥
2
∣
𝑖
≈
𝑓
𝑖
+
1
−
2
𝑓
𝑖
+
𝑓
𝑖
−
1
Δ
𝑥
2
∂x 
2
 
∂ 
2
 f
​
  
​
  
i
​
 ≈ 
Δx 
2
 
f 
i+1
​
 −2f 
i
​
 +f 
i−1
​
 
​
 
在二维情况下：

∇
2
𝑓
≈
𝑓
𝑖
+
1
,
𝑗
+
𝑓
𝑖
−
1
,
𝑗
+
𝑓
𝑖
,
𝑗
+
1
+
𝑓
𝑖
,
𝑗
−
1
−
4
𝑓
𝑖
,
𝑗
Δ
𝑥
2
∇ 
2
 f≈ 
Δx 
2
 
f 
i+1,j
​
 +f 
i−1,j
​
 +f 
i,j+1
​
 +f 
i,j−1
​
 −4f 
i,j
​
 
​
 
✅ 方法类型总结：
类型：中心差分（central difference）

精度：二阶精度

特点：对称、精度高，但在对流占优时容易产生非物理解（建议使用迎风格式或添加耗散）

是否需要我再给你画一个中心差分和迎风差分的比较图？







您说：
import tkinter as tk
from tkinter import ttk, scrolledtext, messagebox, filedialog
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from matplotlib import cm, animation
import matplotlib.contour
import json
import os
import ttkbootstrap as ttk
from ttkbootstrap.constants import *

class CFDSimulationGUI:
    def __init__(self, root):
        self.root = root
        self.root.title("Thermoflow2D Solver Pro")
        # Initialize with light theme
        self.current_theme = 'litera'
        self.style = ttk.Style(theme=self.current_theme)
        # Initialize all tk variables
        self.lx_var = tk.DoubleVar()
        self.ly_var = tk.DoubleVar()
        self.dx_var = tk.DoubleVar()
        self.dy_var = tk.DoubleVar()
        self.re_var = tk.DoubleVar()
        self.dt_var = tk.DoubleVar()
        self.nt_var = tk.IntVar()
        self.pr_var = tk.DoubleVar()
        self.create_widgets()
        self.fig = None
        self.canvas = None
        self.anim = None

    def create_widgets(self):
        # Main frame
        self.main_frame = ttk.Frame(self.root, padding="10", style='primary.TFrame')
        self.main_frame.grid(row=0, column=0, sticky=(tk.W, tk.E, tk.N, tk.S))
        self.root.columnconfigure(0, weight=1)
        self.root.rowconfigure(0, weight=1)

        # Input and plot frames
        self.input_frame = ttk.Frame(self.main_frame, padding="5", style='light.TFrame')
        self.input_frame.grid(row=0, column=0, sticky=(tk.W, tk.N, tk.S), padx=10)
        self.plot_frame = ttk.Frame(self.main_frame, padding="5", style='light.TFrame')
        self.plot_frame.grid(row=0, column=1, sticky=(tk.E, tk.N, tk.S, tk.W), padx=10)
        self.main_frame.columnconfigure(1, weight=3)

        # Notebook
        self.notebook = ttk.Notebook(self.input_frame, style='primary.TNotebook')
        self.notebook.grid(row=0, column=0, sticky=(tk.W, tk.E, tk.N), pady=5)

        # Flow parameters frame
        flow_frame = ttk.Frame(self.notebook, padding="10")
        self.notebook.add(flow_frame, text="Flow Parameters")
        self.create_flow_widgets(flow_frame)

        # Thermal parameters frame
        thermal_frame = ttk.Frame(self.notebook, padding="10")
        self.notebook.add(thermal_frame, text="Thermal Parameters")
        self.create_thermal_widgets(thermal_frame)

        # Console
        self.console = scrolledtext.ScrolledText(
            self.input_frame, height=10, width=50, 
            font=('Arial', 10), wrap=tk.WORD,
            borderwidth=2, relief='flat', 
            background='#f8f9fa' if self.current_theme == 'litera' else '#343a40'
        )
        self.console.grid(row=1, column=0, pady=10, sticky=(tk.W, tk.E))
        self.console.insert(tk.END, "Simulation output will appear here...\n")

        # Button frame
        self.button_frame = ttk.Frame(self.input_frame)
        self.button_frame.grid(row=2, column=0, pady=10, sticky=tk.W)
        ttk.Button(
            self.button_frame, text="Run Simulation", 
            command=self.run_simulation, 
            style='primary.TButton',
            padding=8
        ).grid(row=0, column=0, padx=5)
        ttk.Button(
            self.button_frame, text="Save Config", 
            command=self.save_config, 
            style='success.TButton',
            padding=8
        ).grid(row=0, column=1, padx=5)
        ttk.Button(
            self.button_frame, text="Load Config", 
            command=self.load_config, 
            style='info.TButton',
            padding=8
        ).grid(row=0, column=2, padx=5)
        # Add theme toggle button
        self.theme_button = ttk.Button(
            self.button_frame, text="Switch to Dark Theme", 
            command=self.toggle_theme, 
            style='secondary.TButton',
            padding=8
        )
        self.theme_button.grid(row=0, column=3, padx=5)

        # Plot area label
        self.plot_label = ttk.Label(
            self.plot_frame, 
            text="Animation will appear here after running simulation",
            style='primary.TLabel',
            font=('Arial', 12, 'italic'),
            anchor='center'
        )
        self.plot_label.grid(row=0, column=0, sticky=(tk.N, tk.S, tk.E, tk.W))

    def toggle_theme(self):
        # Toggle between light and dark themes
        if self.current_theme == 'litera':
            self.current_theme = 'darkly'
            self.theme_button.configure(text="Switch to Light Theme")
            self.console.configure(background='#343a40', foreground='#ffffff')
        else:
            self.current_theme = 'litera'
            self.theme_button.configure(text="Switch to Dark Theme")
            self.console.configure(background='#f8f9fa', foreground='#000000')
        
        # Update the style theme
        self.style.theme_use(self.current_theme)
        # Update styles for better visibility in dark mode
        self.style.configure('Custom.TEntry', 
                           fieldbackground='#343a40' if self.current_theme == 'darkly' else '#ffffff',
                           foreground='#ffffff' if self.current_theme == 'darkly' else '#000000')

    def create_flow_widgets(self, frame):
        # Style configuration
        entry_style = ttk.Style()
        entry_style.configure('Custom.TEntry', 
                            padding=5, 
                            font=('Arial', 10),
                            fieldbackground='#343a40' if self.current_theme == 'darkly' else '#ffffff',
                            foreground='#ffffff' if self.current_theme == 'darkly' else '#000000')
        
        params = [
            ("Domain Length (Lx):", self.lx_var, 1.0),
            ("Domain Width (Ly):", self.ly_var, 1.0),
            ("Target dx:", self.dx_var, 0.01),
            ("Target dy:", self.dy_var, 0.01),
            ("Reynolds Number (Re):", self.re_var, 100),
            ("Time Step (dt):", self.dt_var, 0.0005),
            ("Number of Steps (nt):", self.nt_var, 1000)
        ]
        
        for i, (label, var, default) in enumerate(params):
            ttk.Label(frame, text=label, style='primary.TLabel').grid(row=i, column=0, sticky=tk.W, pady=2)
            var.set(default)
            ttk.Entry(frame, textvariable=var, style='Custom.TEntry').grid(row=i, column=1, pady=2, padx=5)

        ttk.Label(frame, text="Velocity Boundary Conditions", style='primary.TLabel', font=('Arial', 11, 'bold')).grid(row=7, column=0, columnspan=2, pady=10)
        
        boundaries = ["left", "right", "top", "bottom"]
        self.u_vars = {}
        self.v_vars = {}
        self.velocity_bc_vars = {}
        
        boundary_labels = {
            "left": "Left",
            "right": "Right",
            "top": "Top",
            "bottom": "Bottom"
        }
        
        for i, boundary in enumerate(boundaries):
            display_label = boundary_labels[boundary]
            ttk.Label(frame, text=f"{display_label} U:").grid(row=8+i, column=0, sticky=tk.W, pady=2)
            self.u_vars[boundary] = tk.DoubleVar(value=1.0 if boundary == "top" else 0.0)
            ttk.Entry(frame, textvariable=self.u_vars[boundary], style='Custom.TEntry').grid(row=8+i, column=1, pady=2)

            ttk.Label(frame, text=f"{display_label} V:").grid(row=8+i, column=2, sticky=tk.W, pady=2)
            self.v_vars[boundary] = tk.DoubleVar(value=0.0)
            ttk.Entry(frame, textvariable=self.v_vars[boundary], style='Custom.TEntry').grid(row=8+i, column=3, pady=2)

            self.velocity_bc_vars[boundary] = tk.BooleanVar(value=False)
            ttk.Checkbutton(
                frame, 
                text=f"{display_label} Neumann", 
                variable=self.velocity_bc_vars[boundary],
                style='primary.TCheckbutton'
            ).grid(row=8+i, column=4, padx=5)

    def create_thermal_widgets(self, frame):
        ttk.Label(frame, text="Prandtl Number (Pr):", style='primary.TLabel').grid(row=0, column=0, sticky=tk.W, pady=2)
        self.pr_var.set(0.71)
        ttk.Entry(frame, textvariable=self.pr_var, style='Custom.TEntry').grid(row=0, column=1, pady=2, padx=5)

        ttk.Label(frame, text="Temperature Boundary Conditions", style='primary.TLabel', font=('Arial', 11, 'bold')).grid(row=1, column=0, columnspan=3, pady=10)
        
        boundaries = ["left", "right", "top", "bottom"]
        self.t_vars = {}
        self.temperature_bc_vars = {}
        
        boundary_labels = {
            "left": "Left",
            "right": "Right",
            "top": "Top",
            "bottom": "Bottom"
        }
        
        default_temps = {
            "left": 300.0,
            "right": 270.0,
            "top": 300.0,
            "bottom": 30.0
        }
        
        for i, boundary in enumerate(boundaries):
            display_label = boundary_labels[boundary]
            ttk.Label(frame, text=f"{display_label} T (K):").grid(row=2+i, column=0, sticky=tk.W, pady=2)
            self.t_vars[boundary] = tk.DoubleVar(value=default_temps[boundary])
            ttk.Entry(frame, textvariable=self.t_vars[boundary], style='Custom.TEntry').grid(row=2+i, column=1, pady=2)

            self.temperature_bc_vars[boundary] = tk.BooleanVar(value=boundary != "top")
            ttk.Checkbutton(
                frame, 
                text=f"{display_label} Neumann", 
                variable=self.temperature_bc_vars[boundary],
                style='primary.TCheckbutton'
            ).grid(row=2+i, column=2, padx=5)

    def save_config(self):
        config = {
            "Lx": self.lx_var.get(),
            "Ly": self.ly_var.get(),
            "dx": self.dx_var.get(),
            "dy": self.dy_var.get(),
            "Re": self.re_var.get(),
            "dt": self.dt_var.get(),
            "nt": self.nt_var.get(),
            "Pr": self.pr_var.get(),
            "velocity": {
                boundary: {
                    "U": self.u_vars[boundary].get(),
                    "V": self.v_vars[boundary].get(),
                    "Neumann": self.velocity_bc_vars[boundary].get()
                } for boundary in ["left", "right", "top", "bottom"]
            },
            "temperature": {
                boundary: {
                    "T": self.t_vars[boundary].get(),
                    "Neumann": self.temperature_bc_vars[boundary].get()
                } for boundary in ["left", "right", "top", "bottom"]
            }
        }
        file_path = filedialog.asksaveasfilename(defaultextension=".json", filetypes=[("JSON files", "*.json")])
        if file_path:
            with open(file_path, "w") as f:
                json.dump(config, f, indent=4)
            self.console.insert(tk.END, f"Configuration saved to {file_path}\n")

    def load_config(self):
        file_path = filedialog.askopenfilename(filetypes=[("JSON files", "*.json")])
        if file_path:
            try:
                with open(file_path, "r") as f:
                    config = json.load(f)
                self.lx_var.set(config["Lx"])
                self.ly_var.set(config["Ly"])
                self.dx_var.set(config["dx"])
                self.dy_var.set(config["dy"])
                self.re_var.set(config["Re"])
                self.dt_var.set(config["dt"])
                self.nt_var.set(config["nt"])
                self.pr_var.set(config["Pr"])
                for boundary in ["left", "right", "top", "bottom"]:
                    self.u_vars[boundary].set(config["velocity"][boundary]["U"])
                    self.v_vars[boundary].set(config["velocity"][boundary]["V"])
                    self.velocity_bc_vars[boundary].set(config["velocity"][boundary]["Neumann"])
                    self.t_vars[boundary].set(config["temperature"][boundary]["T"])
                    self.temperature_bc_vars[boundary].set(config["temperature"][boundary]["Neumann"])
                self.console.insert(tk.END, f"Configuration loaded from {file_path}\n")
            except Exception as e:
                messagebox.showerror("Error", f"Failed to load config: {e}")
                self.console.insert(tk.END, f"Error loading config: {e}\n")

    def run_simulation(self):
        self.console.delete(1.0, tk.END)
        try:
            Lx = self.lx_var.get()
            Ly = self.ly_var.get()
            dx_target = self.dx_var.get()
            dy_target = self.dy_var.get()
            nx = int(np.ceil(Lx / dx_target)) + 1
            ny = int(np.ceil(Ly / dy_target)) + 1
            Re = self.re_var.get()
            dt = self.dt_var.get()
            nt = self.nt_var.get()
            Pr = self.pr_var.get()

            U_left = self.u_vars["left"].get()
            U_right = self.u_vars["right"].get()
            U_top = self.u_vars["top"].get()
            U_bottom = self.u_vars["bottom"].get()
            V_left = self.v_vars["left"].get()
            V_right = self.v_vars["right"].get()
            V_top = self.v_vars["top"].get()
            V_bottom = self.v_vars["bottom"].get()
            velocity_bc_left = self.velocity_bc_vars["left"].get()
            velocity_bc_right = self.velocity_bc_vars["right"].get()
            velocity_bc_top = self.velocity_bc_vars["top"].get()
            velocity_bc_bottom = self.velocity_bc_vars["bottom"].get()

            T_left = self.t_vars["left"].get()
            T_right = self.t_vars["right"].get()
            T_top = self.t_vars["top"].get()
            T_bottom = self.t_vars["bottom"].get()
            temperature_bc_left = self.temperature_bc_vars["left"].get()
            temperature_bc_right = self.temperature_bc_vars["right"].get()
            temperature_bc_top = self.temperature_bc_vars["top"].get()
            temperature_bc_bottom = self.temperature_bc_vars["bottom"].get()

            U_default = 1.0
            velocities = [U_left, U_right, U_top, U_bottom, V_left, V_right, V_top, V_bottom]
            non_zero_velocities = [v for v in velocities if v != 0]
            if non_zero_velocities:
                U = non_zero_velocities[0]
                self.console.insert(tk.END, f'Using velocity {U} to compute viscosity\n')
            else:
                U = U_default
                self.console.insert(tk.END, f'Warning: All input velocities are zero, using default velocity {U_default}\n')

            nu = U * Ly / Re
            dx = Lx / (nx - 1)
            dy = Ly / (ny - 1)
            alpha = nu / Pr

            U_max = max(abs(v) for v in velocities) or U_default
            dt_adv = 0.5 * min(dx, dy) / U_max
            dt_diff = 0.25 * min(dx*dx, dy*dy) / max(nu, alpha)
            dt_safe = min(dt_adv, dt_diff)

        
            self.console.insert(tk.END, f'Reynolds Number = {Re}\n')
            self.console.insert(tk.END, f'Prandtl Number = {Pr}\n')
            self.console.insert(tk.END, f'Current Time Step = {dt}\n')
            self.console.insert(tk.END, f'Safe Time Step <= {dt_safe}\n')
            self.console.insert(tk.END, f'Fluid Viscosity = {nu}\n')
            self.console.insert(tk.END, f'Thermal Diffusivity = {alpha}\n')
            self.console.insert(tk.END, f'Grid Size nx = {nx}, ny = {ny}\n')
            self.console.insert(tk.END, f'Actual Grid Spacing dx = {dx}, dy = {dy}\n')

            omega = np.zeros((ny, nx))
            psi = np.zeros((ny, nx))
            u_full = np.zeros((ny, nx))
            v_full = np.zeros((ny, nx))
            T = np.ones((ny, nx)) * T_bottom

            skip = max(1, nt // 200)
            psi_hist, omega_hist, u_hist, v_hist, T_hist = [], [], [], [], []

            coefdx = dx**2 / (2 * (dx**2 + dy**2))
            coefdy = dy**2 / (2 * (dx**2 + dy**2))
            coefdxdy = dx**2 * dy**2 / (2 * (dx**2 + dy**2))

            def apply_velocity_bc(psi, omega, u_full, v_full):
                y = np.linspace(0, Ly, ny)
                psi[:, 0] = U_left * y
                psi[:, -1] = psi[:, -2]
                psi[0, :] = 0.0
                psi[-1, :] = U_left * Ly
                if velocity_bc_left:
                    u_full[:, 0] = u_full[:, 1]
                    v_full[:, 0] = v_full[:, 1]
                else:
                    u_full[:, 0] = U_left
                    v_full[:, 0] = U_left
                if velocity_bc_right:
                    u_full[:, -1] = u_full[:, -2]
                    v_full[:, -1] = v_full[:, -2]
                else:
                    u_full[:, -1] = U_right
                    v_full[:, -1] = V_right
                if velocity_bc_bottom:
                    u_full[0, :] = u_full[1, :]
                    v_full[0, :] = v_full[1, :]
                else:
                    u_full[0, :] = U_bottom
                    v_full[0, :] = V_bottom
                if velocity_bc_top:
                    u_full[-1, :] = u_full[-2, :]
                    v_full[-1, :] = v_full[-2, :]
                else:
                    u_full[-1, :] = U_top
                    v_full[-1, :] = V_top
                omega[:, 0] = 0.0
                omega[:, -1] = omega[:, -2]
                omega[0, :] = (psi[0, :] - psi[1, :]) * 2.0 / dy**2 - 2.0 * u_full[0, :] / dy
                omega[-1, :] = (psi[-1, :] - psi[-2, :]) * 2.0 / dy**2 - 2.0 * u_full[-1, :] / dy
                return psi, omega, u_full, v_full

            def apply_temperature_bc(T):
                if temperature_bc_top:
                    T[-1, :] = T[-2, :]
                else:
                    T[-1, :] = T_top
                if temperature_bc_bottom:
                    T[0, :] = T[1, :]
                else:
                    T[0, :] = T_bottom
                if temperature_bc_left:
                    T[:, 0] = T[:, 1]
                else:
                    T[:, 0] = T_left
                if temperature_bc_right:
                    T[:, -1] = T[:, -2]
                else:
                    T[:, -1] = T_right
                return T

            for p in range(nt):
                if p > 0:
                    psi, omega, u_full, v_full = apply_velocity_bc(psi, omega, u_full, v_full)
                    px = -(psi[2:,1:-1] - psi[:-2,1:-1]) * (omega[1:-1,2:] - omega[1:-1,:-2]) / (4*dy*dx)
                    py = (omega[2:,1:-1] - omega[:-2,1:-1]) * (psi[1:-1,2:] - psi[1:-1,:-2]) / (4*dy*dx)
                    pxy = (omega[1:-1,2:] + omega[1:-1,:-2] - 2*omega[1:-1,1:-1]) / dx**2 \
                        + (omega[2:,1:-1] + omega[:-2,1:-1] - 2*omega[1:-1,1:-1]) / dy**2
                    omega[1:-1,1:-1] += dt * (px + py + nu * pxy)
                    err = 1.0
                    counter = 0
                    while err > 1e-5 and counter < 200:
                        psi_old = psi.copy()
                        psi[1:-1,1:-1] = coefdxdy * omega[1:-1,1:-1] \
                                      + coefdx * (psi[2:,1:-1] + psi[:-2,1:-1]) \
                                      + coefdy * (psi[1:-1,2:] + psi[1:-1,:-2])
                        err = np.linalg.norm(psi - psi_old, ord=np.inf)
                        counter += 1
                    u_full[1:-1,1:-1] = (psi[2:,1:-1] - psi[:-2,1:-1]) / (2*dy)
                    v_full[1:-1, 1:-1] = -(psi[1:-1, 2:] - psi[1:-1, :-2]) / (2*dx)
                    psi, omega, u_full, v_full = apply_velocity_bc(psi, omega, u_full, v_full)
                    T = apply_temperature_bc(T)
                    px = -(psi[2:,1:-1] - psi[:-2,1:-1]) * (T[1:-1,2:] - T[1:-1,:-2]) / (4*dy*dx)
                    py = (T[2:,1:-1] - T[:-2,1:-1]) * (psi[1:-1,2:] - psi[1:-1,:-2]) / (4*dy*dx)
                    pxy = (T[1:-1,2:] + T[1:-1,:-2] - 2*T[1:-1,1:-1]) / dx**2 \
                        + (T[2:,1:-1] + T[:-2,1:-1] - 2*T[1:-1,1:-1]) / dy**2
                    T[1:-1,1:-1] += dt * (px + py + alpha * pxy)
                if p % skip == 0:
                    psi_hist.append(psi.copy())
                    omega_hist.append(omega.copy())
                    u_hist.append(u_full.copy())
                    v_hist.append(v_full.copy())
                    T_hist.append(T.copy())

            x = np.linspace(0, Lx, nx)
            y = np.linspace(0, Ly, ny)
            X, Y = np.meshgrid(x, y)
            y_slice = Ly / 2
            j_slice = int(np.round(y_slice / Ly * (ny - 1)))

            if self.canvas is not None:
                self.canvas.get_tk_widget().destroy()
            if self.fig is not None:
                plt.close(self.fig)
            if self.anim is not None:
                self.anim.event_source.stop()

            self.fig, axes = plt.subplots(2, 3, figsize=(12, 8), dpi=100)
            self.fig.patch.set_facecolor('#f8f9fa' if self.current_theme == 'litera' else '#343a40')
            axes = axes.flatten()

            plt.subplots_adjust(wspace=0.4, hspace=0.5)

            plots = {
                'mag': {'ax': axes[0], 'contour': None, 'stream': None,
                        'title': 'Velocity Magnitude and Streamlines',
                        'levels': np.arange(0, U_max + 0.1, 0.1),
                        'cbar_label': 'Velocity Magnitude (m/s)'},
                'u': {'ax': axes[1], 'contour': None,
                      'title': 'Horizontal Velocity (u)',
                      'levels': np.arange(-U_max*0.5, U_max + 0.1, 0.1),
                      'cbar_label': 'u-Velocity (m/s)'},
                'v': {'ax': axes[2], 'contour': None,
                      'title': 'Vertical Velocity (v)',
                      'levels': np.arange(-U_max*0.3, U_max*0.3 + 0.01, 0.03),
                      'cbar_label': 'v-Velocity (m/s)'},
                'T': {'ax': axes[3], 'contour': None,
                      'title': 'Temperature Field',
                      'levels': np.linspace(min(T_bottom, T_left, T_right, T_top),
                                           max(T_bottom, T_left, T_right, T_top), 21),
                      'cbar_label': 'Temperature (K)'},
                'line': {'ax': axes[4], 'lines': None,
                         'title': f'y={y_slice:.2f} Velocity Distribution',
                         'xlabel': 'x (m)', 'ylabel': 'Velocity (m/s)'},
                'T_line': {'ax': axes[5], 'lines': None,
                           'title': f'y={y_slice:.2f} Temperature Distribution',
                           'xlabel': 'x (m)', 'ylabel': 'Temperature (K)'}
            }

            for key in plots:
                ax = plots[key]['ax']
                ax.set_title(plots[key]['title'], fontfamily='Arial', fontsize=10, 
                            color='#000000' if self.current_theme == 'litera' else '#ffffff')
                if key not in ['line', 'T_line']:
                    ax.set_aspect('equal')
                    ax.set_xlabel('x (m)', fontfamily='Arial', 
                                 color='#000000' if self.current_theme == 'litera' else '#ffffff')
                    ax.set_ylabel('y (m)', fontfamily='Arial', 
                                 color='#000000' if self.current_theme == 'litera' else '#ffffff')
                    ax.set_xlim(0, Lx)
                    ax.set_ylim(0, Ly)
                    ax.tick_params(colors='#000000' if self.current_theme == 'litera' else '#ffffff')
                    plots[key]['contour'] = ax.contourf(X, Y, np.zeros_like(X),
                                                        levels=plots[key]['levels'],
                                                        cmap=cm.jet, extend='both')
                    cbar = self.fig.colorbar(plots[key]['contour'], ax=ax,
                                        orientation='vertical',
                                        label=plots[key]['cbar_label'])
                    cbar.ax.yaxis.set_tick_params(
                        color='#000000' if self.current_theme == 'litera' else '#ffffff')
                    cbar.ax.yaxis.set_label_text(
                        plots[key]['cbar_label'], 
                        color='#000000' if self.current_theme == 'litera' else '#ffffff')
                else:
                    ax.set_xlabel(plots[key]['xlabel'], fontfamily='Arial', 
                                 color='#000000' if self.current_theme == 'litera' else '#ffffff')
                    ax.set_ylabel(plots[key]['ylabel'], fontfamily='Arial', 
                                 color='#000000' if self.current_theme == 'litera' else '#ffffff')
                    ax.tick_params(colors='#000000' if self.current_theme == 'litera' else '#ffffff')
                    ax.grid(True, color='#888888')

            def update(frame):
                u = u_hist[frame]
                v = v_hist[frame]
                T = T_hist[frame]
                mag = np.sqrt(u**2 + v**2)
                self.fig.suptitle(f'Time Step: {frame}/{len(u_hist)-1}', y=1.02, fontsize=12, 
                                 fontfamily='Arial', 
                                 color='#000000' if self.current_theme == 'litera' else '#ffffff')

                for key in ['mag', 'u', 'v', 'T']:
                    ax = plots[key]['ax']
                    if plots[key]['contour'] is not None:
                        for artist in ax.collections[:]:
                            if isinstance(artist, matplotlib.contour.QuadContourSet):
                                artist.remove()
                    if key == 'mag':
                        plots[key]['contour'] = ax.contourf(X, Y, mag, levels=plots[key]['levels'], cmap=cm.jet, extend='both')
                    elif key == 'u':
                        plots[key]['contour'] = ax.contourf(X, Y, u, levels=plots[key]['levels'], cmap=cm.jet, extend='both')
                    elif key == 'v':
                        plots[key]['contour'] = ax.contourf(X, Y, v, levels=plots[key]['levels'], cmap=cm.jet, extend='both')
                    elif key == 'T':
                        plots[key]['contour'] = ax.contourf(X, Y, T, levels=plots[key]['levels'], cmap=cm.jet, extend='both')

                if plots['mag']['stream'] is not None:
                    if hasattr(plots['mag']['stream'], 'lines'):
                        plots['mag']['stream'].lines.remove()
                plots['mag']['stream'] = plots['mag']['ax'].streamplot(
                    X, Y, u, v, density=1.0, color='w', linewidth=1.0)

                if plots['line']['lines'] is not None:
                    for line in plots['line']['lines']:
                        line.remove()
                plots['line']['lines'] = []
                plots['line']['lines'].append(plots['line']['ax'].plot(x, u[j_slice, :], 'r-', label='u')[0])
                plots['line']['lines'].append(plots['line']['ax'].plot(x, v[j_slice, :], 'g-', label='v')[0])
                plots['line']['ax'].legend()

                if plots['T_line']['lines'] is not None:
                    for line in plots['T_line']['lines']:
                        line.remove()
                plots['T_line']['lines'] = []
                plots['T_line']['lines'].append(plots['T_line']['ax'].plot(x, T[j_slice, :], 'k-', label='T')[0])
                plots['T_line']['ax'].legend()

                return [plots[key]['contour'] for key in ['mag', 'u', 'v', 'T']] + \
                       [plots['mag']['stream'].lines] + plots['line']['lines'] + plots['T_line']['lines']

            self.plot_label.destroy()
            self.canvas = FigureCanvasTkAgg(self.fig, master=self.root.winfo_children()[0].winfo_children()[1])
            self.canvas.get_tk_widget().grid(row=0, column=0, sticky=(tk.N, tk.S, tk.E, tk.W), padx=5, pady=5)
            self.root.winfo_children()[0].winfo_children()[1].columnconfigure(0, weight=1)
            self.root.winfo_children()[0].winfo_children()[1].rowconfigure(0, weight=1)

            frame_step = max(1, len(u_hist) // 40)
            self.anim = animation.FuncAnimation(
                self.fig, update,
                frames=range(0, len(u_hist), frame_step),
                init_func=lambda: update(0),
                interval=200,
                blit=False
            )
            self.canvas.draw()
            self.console.insert(tk.END, "Simulation completed and animation displayed in GUI.\n")

        except Exception as e:
            messagebox.showerror("Error", f"Simulation failed: {e}")
            self.console.insert(tk.END, f"Error during simulation: {e}\n")

if __name__ == "__main__":
    root = ttk.Window()
    app = CFDSimulationGUI(root)
    root.mainloop()这个代码的结构是什么


ChatGPT 说：
你的这段代码是一个 基于 Tkinter + ttkbootstrap 的二维热流体模拟图形界面（GUI）程序，结构非常清晰且模块化。它结合了用户输入接口、计算内核（Navier-Stokes + 热传导） 和 动画可视化 三大核心功能，具有良好的可拓展性。

🔧 代码结构总览：
plaintext
复制
编辑
Thermoflow2D Solver Pro
├── 1. GUI 主框架
│   ├── Tkinter 主窗口 root
│   └── CFDSimulationGUI 类
│       ├── create_widgets()       # 初始化GUI布局（左右分区、标签、按钮）
│       ├── create_flow_widgets()  # 创建流体参数设置框
│       ├── create_thermal_widgets() # 创建热参数设置框
│       ├── toggle_theme()         # 明暗主题切换
│       ├── save_config() / load_config()  # 配置保存与读取（JSON）
│       └── run_simulation()       # 主计算入口
│           ├── 初始化网格与参数
│           ├── apply_velocity_bc()     # 应用速度边界条件
│           ├── apply_temperature_bc()  # 应用温度边界条件
│           ├── 时间推进循环（Navier-Stokes + 温度）
│           └── 动画创建与嵌入GUI
└── __main__ 启动入口
🧩 功能分层模块说明：
1️⃣ 用户界面（GUI 控件）
使用 ttkbootstrap 提供现代化主题（如 litera, darkly）

提供 输入参数框（Entry）+ Neumann边界选择框（Checkbutton）

提供保存 / 读取配置按钮（绑定 json 文件）

控制台输出：用于实时显示模拟参数和状态

2️⃣ 数值计算模块（run_simulation）
构造二维网格 (nx, ny)

解不可压缩二维Navier-Stokes 方程组（涡度–流函数法）

解能量守恒方程（显式欧拉 + 中心差分）

使用 SOR 方法求解泊松方程 
∇
2
𝜓
=
−
𝜔
∇ 
2
 ψ=−ω

3️⃣ 可视化模块
使用 matplotlib 生成 6 子图（速度、温度、剖面图等）

嵌入 Tkinter GUI via FigureCanvasTkAgg

使用 FuncAnimation 实时更新图像动画帧

🔄 数值方法小结
方程	方法	时间离散	空间离散
涡度方程	显式欧拉	向前差分	二阶中心差分
流函数泊松方程	SOR 迭代	伪稳态	5点差分模板
热传导方程	显式欧拉	向前差分	二阶中心差分

✅ 优点：
模块清晰，易于维护和扩展；

支持 GUI 交互、配置存取和主题切换；

实现了典型 CFD 中最基础的显式涡度-流函数方法。

如果你需要进一步封装成独立软件、加速计算（如Numba/Cython），或添加更高阶数值方法（如Runge-Kutta, Crank-Nicolson），我可以继续协助优化设计。需要我进一步分析哪一块模块或添加哪些功能？







您说：
根据这个代码给我写一个readme文件。我的文件夹结构是一个GUI.py文件，一个User Manual


ChatGPT 说：
以下是为你的项目 Thermoflow2D Solver Pro 准备的专业 README.md 文件，适用于 GitHub 或本地展示：

markdown
复制
编辑
# 🌬️ Thermoflow2D Solver Pro

A 2D laminar fluid flow and heat transfer simulation tool with a modern graphical user interface (GUI), built using `Tkinter` and `ttkbootstrap`. This solver uses the vorticity-streamfunction formulation of the incompressible Navier-Stokes equations, coupled with the energy equation.

---

## 📁 Folder Structure

Thermoflow2D-Solver-Pro/
├── GUI.py # Main Python GUI application
└── User Manual.pdf # User guide with instructions and screenshots

yaml
复制
编辑

---

## 💡 Features

- ✅ Interactive GUI with real-time simulation controls
- 🎨 Light/Dark theme toggle via `ttkbootstrap`
- 💾 Save and load simulation configurations (`.json`)
- 📈 Built-in animation of velocity fields and temperature distribution
- 📊 Line plots of horizontal/vertical velocities and temperature profiles
- 🧮 Solves:
  - 2D incompressible Navier-Stokes (vorticity-streamfunction)
  - 2D energy equation (heat transfer)

---

## 🔧 Numerical Methods

| Component           | Method                |
|---------------------|------------------------|
| Time Integration    | Explicit Euler         |
| Spatial Derivatives | Second-order Central Difference |
| Poisson Solver      | Successive Over-Relaxation (SOR) |
| Advection-Diffusion | Explicit scheme        |

---

## ▶️ How to Run

### 🐍 Requirements

- Python ≥ 3.8
- Packages:
  - `numpy`
  - `matplotlib`
  - `ttkbootstrap`

### 💻 Run the GUI

```bash
python GUI.py
📘 User Manual
See User Manual.pdf for step-by-step instructions including:

How to set up flow and thermal parameters

How to interpret the plots

How to use Neumann/Dirichlet boundary conditions

How to save/load configurations
🖼️ Interface Preview
Animation window with 6 subplots:

Velocity magnitude & streamlines

u / v component plots

Temperature field

Midline velocity & temperature profiles
