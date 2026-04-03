#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Simple tool to see how gradient changes with source position as in Fig. 3
"""
# Imports:
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
from magnetpy.paper_figs._paths import example_data_path

outpath = example_data_path("paper_figs")
outpath.mkdir(parents=True, exist_ok=True)

# -----------------------------
# Geometry parameters
# -----------------------------
d = 5.0   # horizontal spacing (x)
h = d/np.sqrt(2)   # vertical spacing (z)

# -----------------------------
# Sensor positions (x, y, z)
# -----------------------------
sensor_pos = np.array([
    [-d, 0.0, 0.0],        # Sensor 1
    [d,   0.0, 0.0],        # Sensor 2
    [0.0,   d, h],    # Sensor 3
    [0.0, -d, h] # Sensor 4
])

# Center of array
array_center = sensor_pos.mean(axis=0)

# -----------------------------
# Source positions
# -----------------------------
source_t0 = np.array([-20.0, -10.5, 15.0])
source_t1 = source_t0 + np.array([-50.0, -20.0, -40.0])  # shift

# -----------------------------
# Vector computation
# -----------------------------
def vectors_to_source(sensors, source, scale=2.0):
    vecs = source - sensors
    norms = np.linalg.norm(vecs, axis=1, keepdims=True)
    vecs = vecs / norms  # unit vectors
    return scale * vecs

vectors_t0 = vectors_to_source(sensor_pos, source_t0)
vectors_t1 = vectors_to_source(sensor_pos, source_t1)

# Change in vectors
delta_vectors = vectors_t1 - vectors_t0

# Average vectors
avg_v0 = vectors_t0.mean(axis=0)
avg_v1 = vectors_t1.mean(axis=0)

# -----------------------------
# Plot
# -----------------------------
fig = plt.figure(figsize=(10, 7))

# ---------- Subplot 1: t = 0 ----------
ax1 = fig.add_subplot(1, 2, 1, projection='3d')

# Sensor positions
ax1.scatter(*sensor_pos.T, s=70, color='black')
# ax1.scatter(*source_t0, s=120, color='crimson', marker='*')

# Sensor labels
for i, (x, y, z) in enumerate(sensor_pos):
    ax1.text(x, y, z + 0.3, f"{i+1}", fontsize=12, weight='bold')

# Per-sensor vector
ax1.quiver(
    sensor_pos[:, 0], sensor_pos[:, 1], sensor_pos[:, 2],
    vectors_t0[:, 0], vectors_t0[:, 1], vectors_t0[:, 2],
    color='royalblue', linewidth=2, arrow_length_ratio=0.5
)

# Average vector
ax1.quiver(
    array_center[0], array_center[1], array_center[2],
    avg_v0[0], avg_v0[1], avg_v0[2],
    color='purple', linewidth=3, arrow_length_ratio=0.5, pivot='tail'
)

ax1.set_title("t = 0", fontsize=20, y=0.97)
ax1.set_xlabel("X [cm]", fontsize=10)
ax1.set_ylabel("Y [cm]", fontsize=10)
ax1.set_zlabel("Z [cm]", fontsize=10)
ax1.set_box_aspect([1, 1, 1])

# ---------- Subplot 2: t = 1 + Δv ----------
ax2 = fig.add_subplot(1, 2, 2, projection='3d')

# Sensor positions
ax2.scatter(*sensor_pos.T, s=70, color='black')
# ax2.scatter(*source_t1, s=120, color='crimson', marker='*')

# Sensor labels
for i, (x, y, z) in enumerate(sensor_pos):
    ax2.text(x, y, z + 0.3, f"{i+1}", fontsize=12, weight='bold')

# Per-sensor vector at t = 1
v1 = ax2.quiver(
    sensor_pos[:, 0], sensor_pos[:, 1], sensor_pos[:, 2],
    vectors_t1[:, 0], vectors_t1[:, 1], vectors_t1[:, 2],
    color='royalblue', linewidth=2, arrow_length_ratio=0.5,
    label=r'$\nabla \mathbf{B}$', pivot='tail'
)

# Change vector Δv
dv = ax2.quiver(
    sensor_pos[:, 0], sensor_pos[:, 1], sensor_pos[:, 2],
    delta_vectors[:, 0], delta_vectors[:, 1], delta_vectors[:, 2],
    color='firebrick', linewidth=2, arrow_length_ratio=0.5,
    label=r'$\Delta (\nabla \mathbf{B})$', pivot='tail'
)

# Average vector
avg = ax2.quiver(
    array_center[0], array_center[1], array_center[2],
    avg_v1[0], avg_v1[1], avg_v1[2],
    color='purple', linewidth=3, arrow_length_ratio=0.5,
    label=r'$\langle \nabla \mathbf{B} \rangle$', pivot='tail'
)

ax2.set_title("t = 1", fontsize=20, y=0.97)
ax2.set_xlabel("X [cm]", fontsize=10)
ax2.set_ylabel("Y [cm]", fontsize=10)
ax2.set_zlabel("Z [cm]", fontsize=10)
ax2.set_box_aspect([1, 1, 1])

leg = fig.legend(
    handles=[v1, avg, dv],
    loc='lower center',
    ncol=3,
    frameon=False,
    bbox_to_anchor=(0.5, 0.02),
    fontsize=18
)
for line in leg.get_lines():
    line.set_linewidth(5.0)

fig.suptitle(r"Change in $\nabla \mathbf{B}$ between timesteps", y=0.9, fontsize=24)

# Common limits
for ax in (ax1, ax2):
    ax.set_xlim(-12, 12)
    ax.set_ylim(-12, 12)
    ax.set_zlim(-1, h + 2)

    ax.view_init(elev=20., azim=-68.)

plt.tight_layout(rect=[-0.05, 0.075, 0.98, 0.95])
out_png = outpath / "sensor_vector_change.png"
fig.savefig(out_png, dpi=200, bbox_inches='tight')
print(f"Wrote PNG: {out_png}")
plt.show()
