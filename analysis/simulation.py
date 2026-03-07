import numpy as np
import matplotlib.pyplot as plt

DT=0.1
STEPS=300
V=1.0
OMEGA=0.2

ODOM_V_STD=0.05
ODOM_OMEGA_STD=0.02

np.random.seed(42)

def motion_model(state,v,omega,dt):
    x,y,theta=state
    x+=v*np.cos(theta)*dt
    y+=v*np.sin(theta)*dt
    theta+=omega*dt
    return np.array([x,y,theta])

# 1. Ground Truth

ground_truth=[np.array([0.0,0.0,0.0])]

for _ in range(STEPS):
    new_state=motion_model(ground_truth[-1],V,OMEGA,DT)
    ground_truth.append(new_state)

ground_truth=np.array(ground_truth)

# 2. Noisy Odometry

odometry=[np.array([0.0,0.0,0.0])]

for _ in range(STEPS):
    v_noisy=V+np.random.normal(0,ODOM_V_STD)
    omega_noisy=OMEGA+np.random.normal(0,ODOM_OMEGA_STD)
    new_state=motion_model(odometry[-1],v_noisy,omega_noisy,DT)
    odometry.append(new_state)

odometry=np.array(odometry)

# 3. Landmarks

landmarks=np.array([
    [3.0,2.0],
    [6.0,4.0],
    [2.0,7.0],
    [8.0,1.0],
    [5.0,8.0],
])

# 4. Observations (range + bearing)

RANGE_STD=0.2
BEARING_STD=0.05
MAX_RANGE=5.0

observations=[]

for step_i,state in enumerate(ground_truth):
    x,y,theta=state
    for lm_id,(lx,ly) in enumerate(landmarks):
        dx=lx-x
        dy=ly-y

        true_range=np.sqrt(dx**2+dy**2)
        true_bearing=np.arctan2(dy,dx)-theta

        if true_range<=MAX_RANGE:
            noisy_range=true_range+np.random.normal(0,RANGE_STD)
            noisy_bearing=true_bearing+np.random.normal(0,BEARING_STD)
            observations.append((step_i,lm_id,noisy_range,noisy_bearing))

observations=np.array(observations)

# 5. Save Data

np.save("ground_truth.npy",ground_truth)
np.save("odometry.npy",odometry)
np.save("landmarks.npy",landmarks)
np.save("observations.npy",observations)

print(f"ground_truth shape:{ground_truth.shape}")
print(f"odometry shape:{odometry.shape}")
print(f"landmarks shape:{landmarks.shape}")
print(f"observations shape:{observations.shape}")
print(f"total observations:{len(observations)}")

# 6. Plot

plt.figure(figsize=(8,8))

plt.plot(ground_truth[:,0],ground_truth[:,1],'g-',linewidth=2,label='Ground Truth')
plt.plot(odometry[:,0],odometry[:,1],'r--',linewidth=1.5,label='Noisy Odometry')
plt.scatter(landmarks[:,0],landmarks[:,1],marker='*',s=200,c='blue',zorder=5,label='Landmarks')
plt.scatter([0],[0],marker='o',s=100,c='black',zorder=6,label='Start')

plt.legend()
plt.title('Simulated Robot Trajectory')
plt.xlabel('x(m)')
plt.ylabel('y(m)')
plt.axis('equal')
plt.grid(True)
plt.tight_layout()
plt.savefig("trajectory_plot.png",dpi=150)
plt.show()

print("plot saved to trajectory_plot.png")
