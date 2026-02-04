import numpy as np
import plotly.graph_objects as go

def visualize_disk_method_25d(x_points=[1, 1.5, 2]):
    """
    Visualizes the Disk Method for y = sqrt(x) in a 2.5D Isometric view.
    Includes buttons to toggle between Rotating Points (Disks) and Rotating Function (Solid).
    
    Parameters:
    - x_points: List of x-values to visualize as rotating disks.
    """
    # 1. Setup Data
    x_range = np.linspace(0, 4, 100)
    y_curve = np.sqrt(x_range)
    z_curve = np.zeros_like(x_range)
    
    # 2. Initialize Figure
    fig = go.Figure()

    # --- TRACE 0: Axis of Rotation (X-Axis) ---
    fig.add_trace(go.Scatter3d(
        x=[-0.5, 5], y=[0, 0], z=[0, 0],
        mode='lines',
        line=dict(color='black', width=3),
        name='Axis of Rotation'
    ))

    # --- TRACE 1: The Function Line (y = sqrt(x)) ---
    # Swapped y and z: Curve now lies on the X-Z plane
    fig.add_trace(go.Scatter3d(
        x=x_range, y=z_curve, z=y_curve,
        mode='lines', 
        line=dict(color='blue', width=5),
        name='y=√x'
    ))

    # --- TRACE 2: The Selected Points (Start Position) ---
    xp = np.array(x_points)
    yp = np.sqrt(xp)
    zp = np.zeros_like(xp)
    
    # Trace for Points (Initially Visible)
    fig.add_trace(go.Scatter3d(
        x=xp, y=zp, z=yp,
        mode='markers', 
        marker=dict(size=6, color='red', symbol='circle'),
        name='Sample Points'
    ))

    # --- TRACES 3 to (3 + 2*N): The Disks (Rim + Face per point) ---
    # We initialize them as empty placeholders to be filled by animation
    for x_val in x_points:
        # Rim Trace
        fig.add_trace(go.Scatter3d(
            x=[], y=[], z=[],
            mode='lines', line=dict(color='red', width=3),
            name=f'Disk Rim x={x_val}', showlegend=False
        ))
        # Face Trace
        fig.add_trace(go.Mesh3d(
            x=[], y=[], z=[],
            color='red', opacity=0.5, showscale=False,
            name=f'Disk Face x={x_val}'
        ))

    # --- TRACE Last: The Solid Surface (Placeholder) ---
    # Initially hidden (visible=False)
    fig.add_trace(go.Surface(
        x=[], y=[], z=[],
        colorscale='Viridis', opacity=0.8, showscale=False,
        name='Swept Volume',
        visible=False 
    ))

    # --- CALCULATE INDICES FOR VISIBILITY TOGGLES ---
    # 0: Axis, 1: Curve, 2: Points
    # 3..End-1: Disks (2 traces per point)
    # End: Surface
    num_disk_traces = 2 * len(x_points)
    total_traces = 3 + num_disk_traces + 1
    
    # Visibility Arrays
    # Mode A (Disks): True for Axis, Curve, Points, Disks. False for Surface.
    vis_disks = [True] * (total_traces - 1) + [False]
    
    # Mode B (Solid): True for Axis(0), Curve(1), Surface(Last). False for Points(2) and Disks.
    vis_solid = [True, True] + [False]*(1 + num_disk_traces) + [True]

    # --- ANIMATION FRAMES ---
    frames = []
    steps = np.linspace(0, 2*np.pi, 36) # 36 frames
    
    for k, angle in enumerate(steps):
        frame_data = []
        is_last_frame = (k == len(steps) - 1)
        
        # 1. Axis (Static)
        frame_data.append(go.Scatter3d(x=[-0.5, 5], y=[0, 0], z=[0, 0]))
        # 2. Curve (Static)
        frame_data.append(go.Scatter3d(x=x_range, y=z_curve, z=y_curve))
        
        # 3. Rotating Points (Calculated for every frame)
        y_rot = yp * np.cos(angle)
        z_rot = yp * np.sin(angle)
        frame_data.append(go.Scatter3d(x=xp, y=z_rot, z=y_rot)) # Swapped y/z
        
        # 4. Disks (Calculated for every frame)
        for x_val in x_points:
            y_rad = np.sqrt(x_val)
            num_points = max(2, int(30 * (angle/(2*np.pi))) + 2)
            theta_slice = np.linspace(0, angle, num_points)
            
            y_c = y_rad * np.cos(theta_slice)
            z_c = y_rad * np.sin(theta_slice)
            x_c = np.full_like(theta_slice, x_val)
            
            # Rim (Swapped y/z)
            frame_data.append(go.Scatter3d(x=x_c, y=z_c, z=y_c))
            
            # Face (Mesh)
            x_mesh = np.concatenate(([x_val], x_c))
            y_mesh = np.concatenate(([0], z_c)) # Swapped
            z_mesh = np.concatenate(([0], y_c)) # Swapped
            
            M = len(theta_slice)
            if M >= 2:
                i_idx = [0] * (M - 1)
                j_idx = list(range(1, M))
                k_idx = list(range(2, M + 1))
            else:
                i_idx, j_idx, k_idx = [], [], []
                
            frame_data.append(go.Mesh3d(
                x=x_mesh, y=y_mesh, z=z_mesh,
                i=i_idx, j=j_idx, k=k_idx,
                color='red', opacity=0.5
            ))
            
        # 5. Solid Surface (Calculated for every frame)
        if angle > 0:
            theta_slice = np.linspace(0, angle, 30)
            X_S, T_S = np.meshgrid(x_range, theta_slice)
            R = np.sqrt(X_S)
            
            # Math coords
            Y_math = R * np.cos(T_S)
            Z_math = R * np.sin(T_S)
            
            # Plot coords (Swapped y/z)
            frame_data.append(go.Surface(
                x=X_S, y=Z_math, z=Y_math,
                colorscale='Viridis', opacity=0.8
            ))
        else:
            frame_data.append(go.Surface(x=[], y=[], z=[]))
            
        frames.append(go.Frame(data=frame_data, name=f'rot{k}'))

    fig.frames = frames

    # 4. 2.5D Layout Configuration & Buttons
    updatemenus = [dict(
        type="buttons",
        direction="left",
        x=0.0, y=1.1, # Position buttons above the plot
        xanchor='left', yanchor='top',
        buttons=list([
            dict(
                label="Show Disks",
                method="update",
                args=[{"visible": vis_disks},
                      {"title": "Disk Method: Rotating Points"}]
            ),
            dict(
                label="Show Solid",
                method="update",
                args=[{"visible": vis_solid},
                      {"title": "Disk Method: Full Volume"}]
            )
        ]),
    )]

    fig.update_layout(
        title="Disk Method",
        width=800,
        height=600,
        updatemenus=updatemenus,
        scene=dict(
            xaxis=dict(title='X (Axis of Rotation)', range=[0, 4.5], showgrid=True),
            # Labels removed for Y and Z (tick labels removed)
            yaxis=dict(title='', range=[-2.5, 2.5], showgrid=True, showticklabels=False),
            zaxis=dict(title='', range=[-2.5, 2.5], showgrid=True, showticklabels=False),
            aspectmode='cube', # Forces equal unit scaling
            
            # --- THE MAGIC 2.5D SETTINGS ---
            camera=dict(
                projection=dict(type="orthographic"), # Flattens depth
                eye=dict(x=1.5, y=1.5, z=1.5),       # Isometric Angle
                up=dict(x=0, y=0, z=1)
            )
        ),
        sliders=[dict(
            steps=[dict(method='animate', 
                        args=[[f'rot{k}'], dict(mode='immediate', frame=dict(duration=20, redraw=True))],
                        label='') for k in range(len(steps))], 
            currentvalue=dict(visible=False)
        )]
    )

    fig.show()

# Run the 2.5D visualization
visualize_disk_method_25d(x_points=[1, 2, 3])