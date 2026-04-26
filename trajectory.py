import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import os

# --- CONFIGURATION ---
L1 = 1.0  # Length of link 1
L2 = 1.0  # Length of link 2
FPS = 20  # Matches your DT=0.05 (1/0.05 = 20)

def select_csv(base_dir="saved_models"):
    """Finds all rollout CSVs and lets the user choose one."""
    csv_files = []
    # Walk through the directory to find all relevant CSVs
    for root, dirs, files in os.walk(base_dir):
        for file in files:
            # Look for any rollout CSV files
            if file.endswith(".csv") and "rollout" in file:
                csv_files.append(os.path.join(root, file))
    
    if not csv_files:
        raise FileNotFoundError(f"No rollout CSVs found in '{base_dir}'.")
    
    # Sort by modification time (newest first)
    csv_files.sort(key=os.path.getmtime, reverse=True)
    
    print("\n--- Available Trajectories ---")
    for i, f in enumerate(csv_files):
        # Highlighting the specific files you mentioned if they exist
        filename = os.path.basename(f)
        print(f" [{i}] {filename}")
    
    try:
        selection = input(f"\nSelect file number (0-{len(csv_files)-1}, default 0): ").strip()
        idx = int(selection) if selection.isdigit() else 0
        if idx < 0 or idx >= len(csv_files):
            print("Invalid index, defaulting to 0.")
            idx = 0
    except ValueError:
        idx = 0
        
    return csv_files[idx]

def main():
    try:
        csv_path = select_csv()
    except FileNotFoundError as e:
        print(e)
        return

    df = pd.read_csv(csv_path)

    # 1. KINEMATICS
    q1 = df['q1_rad'].values
    q2 = df['q2_rad'].values

    # Joint 1 (Pivot to Link 1 end)
    x1 = L1 * np.sin(q1)
    y1 = -L1 * np.cos(q1)

    # Joint 2 (Link 1 end to Link 2 end)
    x2 = x1 + L2 * np.sin(q1 + q2)
    y2 = y1 - L2 * np.cos(q1 + q2)

    # 3. SETUP PLOT (Adjusted for warm/low-brightness preference)
    plt.style.use('dark_background')
    fig = plt.figure(figsize=(8, 8))
    ax = fig.add_subplot(111, aspect='equal', autoscale_on=False, xlim=(-2.2, 2.2), ylim=(-2.2, 2.2))
    
    # Use a softer gray for the grid to reduce eye strain
    ax.grid(True, color='#333333', linestyle='--', linewidth=0.5)

    # Elements to animate - Using softer, less "neon" colors
    line, = ax.plot([], [], 'o-', lw=3, color='#5c7bad', markersize=7, markerfacecolor='#dddddd')
    trace, = ax.plot([], [], '-', lw=1, color='#a65c5c', alpha=0.4)
    time_text = ax.text(0.05, 0.9, '', transform=ax.transAxes, color='#bbbbbb')

    history_x, history_y = [], []

    def init():
        line.set_data([], [])
        trace.set_data([], [])
        time_text.set_text('')
        return line, trace, time_text

    def animate(i):
        thisx = [0, x1[i], x2[i]]
        thisy = [0, y1[i], y2[i]]

        history_x.append(x2[i])
        history_y.append(y2[i])
        
        # Keep trace limited to last 50 frames
        if len(history_x) > 50:
            history_x.pop(0)
            history_y.pop(0)

        line.set_data(thisx, thisy)
        trace.set_data(history_x, history_y)
        
        # Check if columns exist before printing to avoid errors
        time_val = df['time_s'][i] if 'time_s' in df.columns else i/FPS
        goal_val = df['goal_dist'][i] if 'goal_dist' in df.columns else 0.0
        
        time_text.set_text(f"Time: {time_val:.2f}s\nGoal Dist: {goal_val:.2f}")
        return line, trace, time_text

    ani = animation.FuncAnimation(fig, animate, frames=len(df),
                                  interval=1000/FPS, blit=True, init_func=init)

    print(f"Animating: {os.path.basename(csv_path)}")
    plt.show()

if __name__ == "__main__":
    main()