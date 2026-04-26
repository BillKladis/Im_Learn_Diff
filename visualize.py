import os
import json
import argparse
import math
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

# Standard baseline for the main head angles [q1, q2]
BASE_Q1 = 200.0
BASE_Q2 = 50.0 # Note: user set Qf as [200, 200, 50, 50] in their file

def select_run_dir(base_dir="saved_models", specific_run=None):
    if not os.path.exists(base_dir):
        raise FileNotFoundError(f"Could not find '{base_dir}' directory.")
    if specific_run:
        target_dir = os.path.join(base_dir, specific_run)
        if os.path.exists(target_dir): return target_dir
    subdirs = [d for d in os.listdir(base_dir) if os.path.isdir(os.path.join(base_dir, d))]
    subdirs.sort(key=lambda x: os.path.getmtime(os.path.join(base_dir, x)), reverse=True)
    print("\n--- Available Training Runs ---")
    for i, d in enumerate(subdirs): print(f" [{i}] {d}")
    choice = input("\nEnter the number of the run to plot Qf coupling (default=0): ").strip()
    idx = int(choice) if choice.isdigit() and 0 <= int(choice) < len(subdirs) else 0
    return os.path.join(base_dir, subdirs[idx])

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--run", type=str, default=None)
    args = parser.parse_args()

    run_dir = select_run_dir(specific_run=args.run)
    session_name = os.path.basename(run_dir)
    json_path = os.path.join(run_dir, f"{session_name}_network_outputs_full.json")

    if not os.path.exists(json_path):
        print(f"Error: Could not find full network outputs file at {json_path}")
        return

    print(f"\nLoading and parsing Qf matrices from: {session_name}...")
    with open(json_path, 'r') as f:
        data = json.load(f)
    
    # We only care about the last epoch's behavior
    last_epoch_steps = data['epochs'][-1]['steps']
    
    # We need to compute the elapsed simulation time based on step index
    DT = 0.05
    times = [i * DT for i in range(len(last_epoch_steps))]

    # 1. Extract the key data terms from the Qf_dense matrices
    q1_main_weights = []   # Qf[0,0] (angle penalty)
    q1d_main_weights = []  # Qf[1,1] (velocity penalty)
    coupling_terms = []    # Qf[0,1] (Position-Velocity coupling: momentum reward)
    rotations = []         # Derived rotation angle of the ellipsoid cost valley

    for step_data in last_epoch_steps:
        qf_flat = step_data.get('Qf_dense')
        if qf_flat is None: continue
        
        # qf_flat is likely saved as a flattened 16-element list
        # Format: row0(0,1,2,3), row1(4,5,6,7), ...
        
        w_q1 = qf_flat[0]
        w_q1d = qf_flat[5]
        w_coupling = qf_flat[1] # Q[0,1] or Q[1,0] because it's symmetric

        q1_main_weights.append(w_q1)
        q1d_main_weights.append(w_q1d)
        coupling_terms.append(w_coupling)
        
        # Calculate the rotation angle (in radians) of the principal axis of the ellipsoid
        # Uses the 2D rotation formula for ellipses
        # angle = 0.5 * arctan( 2 * Qxy / (Qx - Qy) )
        delta_diag = w_q1 - w_q1d
        denom = delta_diag if abs(delta_diag) > 1e-9 else 1e-9 # Prevent divide by zero
        angle = 0.5 * math.atan2(2 * w_coupling, denom)
        rotations.append(angle * (180.0 / math.pi)) # Save as degrees

    # Sync time array in case of mismatch
    times = times[:len(coupling_terms)]

    # 2. Plotting the 'Pumping' Logic
    plt.style.use('dark_background')
    fig = plt.figure(figsize=(15, 10), facecolor="#0f0f1a")
    gs = gridspec.GridSpec(3, 2, hspace=0.45, wspace=0.25)
    
    fig.suptitle(f"How the Neural Network is Shaping the Terminal Cost Bowl\n({session_name})", color="white", fontsize=14)

    # Base styling
    def style(ax):
        ax.set_facecolor("#0f0f1a")
        ax.grid(True, alpha=0.15)
        ax.tick_params(colors="#aaaaaa")
        for sp in ax.spines.values(): sp.set_edgecolor("#333355")

    # --- ROW 1: Running Weights vs Velocity ---
    # We need velocity to correlate it against the coupling
    csv_path  = os.path.join(run_dir, f"{session_name}_rollout_final.csv")
    actual_q1d = []
    if os.path.exists(csv_path):
        import csv
        with open(csv_path, 'r') as f:
            reader = csv.DictReader(f)
            actual_q1d = [float(row['q1_dot_rads']) for row in reader]
    actual_q1d = actual_q1d[:len(times)]

    ax_vel = fig.add_subplot(gs[0, 0])
    style(ax_vel)
    ax_vel.plot(times, actual_q1d, color="#44ff88", linewidth=2.5, label="Actual Pendulum Velocity (q1_dot)")
    ax_vel.set_ylabel("Velocity [rad/s]", color="#44ff88")
    ax_vel.set_title("Pendulum Physical Velocity", color="#cccccc")
    ax_vel.grid(True, alpha=0.15)
    
    # --- ROW 2: The Main Coupling Term (WTF is the Momentum Reward doing?) ---
    ax_couple = fig.add_subplot(gs[1, 0])
    style(ax_couple)
    # The moment of truth. If this is 0.0, the network learned nothing.
    ax_couple.plot(times, coupling_terms, color="#ff4466", linewidth=2.5, label=" learned q1-q1_dot Coupling")
    ax_couple.axhline(0.0, color='white', linestyle='--', alpha=0.4)
    ax_couple.set_ylabel("Weight (Coupling)", color="#ff4466")
    ax_couple.set_title("The 'Pumping' Weight: Q[0,1] x (q1-pi) x q1_dot", color="white")
    ax_couple.legend(loc="best")

    # --- ROW 3: Rotational Dynamics ---
    ax_rot = fig.add_subplot(gs[2, 0])
    style(ax_rot)
    ax_rot.plot(times, rotations, color="#ffcc44", linewidth=2.5, label="Ellipsoid Rotation Angle")
    ax_rot.axhline(0.0, color='white', linestyle='--', alpha=0.4)
    ax_rot.set_ylabel("Angle [degrees]", color="#ffcc44")
    ax_rot.set_xlabel("Time [s]")
    ax_rot.set_title("How the network 'tilts' the cost bowl", color="#cccccc")
    
    # --- RIGhT COLUMN: 2D visualizations of the ellipsoid valley ---
    # Draw the shape of the Q bowl at three key moments
    ax_viz = fig.add_subplot(gs[:, 1])
    style(ax_viz)
    ax_viz.set_aspect('equal', 'box')
    ax_viz.set_xlabel("q1 Position Error [rad]")
    ax_viz.set_ylabel("q1_dot Velocity [rad/s]")
    ax_viz.set_title("Visualization of the learned 'Cost Valley'", color="white")
    
    # Setup for drawing ellipses
    theta_viz = np.linspace(0, 2*np.pi, 100)
    
    # Snapshots (start, middle, end)
    indices_to_viz = [0, len(times)//2, len(times)-1]
    # Red to yellow colors
    colors = ["#ff4466", "#ffaa44", "#ffcc44"]
    
    for count, idx in enumerate(indices_to_viz):
        w_q1 = q1_main_weights[idx]
        w_q1d = q1d_main_weights[idx]
        w_coup = coupling_terms[idx]
        time_snap = times[idx]
        
        # Reconstruct the 2x2 part of Qf relevant to q1
        Q = np.array([[w_q1, w_coup], [w_coup, w_q1d]])
        
        # Calculate Eigenvalues/Vectors to get shape and orientation
        evals, evecs = np.linalg.eigh(Q)
        
        # We draw an equi-cost ellipse: x^T Q x = constant
        # Width/height are inversely proportional to sqrt of eigenvalues
        C = 50.0 # arbitrary constant to set visual size
        width = 2.0 * math.sqrt(C / evals[1]) # evals are sorted, [0] is min (the valley), [1] is max (the walls)
        height = 2.0 * math.sqrt(C / evals[0])
        
        # Rotation angle of the vector corresponding to the *minimum* eigenvalue (the valley)
        valley_angle = math.atan2(evecs[1,0], evecs[0,0])
        
        # Draw it
        ellipse_x = (width/2) * np.cos(theta_viz)
        ellipse_y = (height/2) * np.sin(theta_viz)
        
        # Apply rotation
        R_mat = np.array([[math.cos(valley_angle), -math.sin(valley_angle)],
                          [math.sin(valley_angle),  math.cos(valley_angle)]])
        
        rotated = R_mat @ np.vstack([ellipse_x, ellipse_y])
        
        ax_viz.plot(rotated[0,:], rotated[1,:], color=colors[count], 
                    linewidth=2.5, label=f"t={time_snap:.1f}s")
        
        # Draw an arrow showing the principal axis of the "momentum reward"
        ax_viz.arrow(0, 0, height*0.5*math.cos(valley_angle), height*0.5*math.sin(valley_angle),
                     color=colors[count], head_width=0.05, alpha=0.5)

    ax_viz.set_xlim(-1, 1) # Zoom in to see the shape
    ax_viz.set_ylim(-2, 2)
    ax_viz.grid(True, alpha=0.2, linestyle=':')
    ax_viz.axhline(0.0, color='white', linestyle='-', alpha=0.3)
    ax_viz.axvline(0.0, color='white', linestyle='-', alpha=0.3)
    ax_viz.legend(loc="upper right", facecolor="#1a1a2e", edgecolor="#444466")
    
    # Matching background color for save
    fig.patch.set_facecolor('#0f0f1a')
    
    plt.tight_layout()
    save_path = os.path.join(run_dir, f"{session_name}_qf_coupling.png")
    plt.savefig(save_path, dpi=150, facecolor=fig.get_facecolor(), edgecolor='none')
    print(f"Saved coupling visualization to: {save_path}")
    plt.show()

if __name__ == "__main__":
    main()