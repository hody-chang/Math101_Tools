import numpy as np
import plotly.graph_objects as go

def visualize_washer_method_25d(x_points=[0.2, 0.5, 0.8]):
    """
    Visualizes the Washer Method for y = sqrt(x) (outer) and y = x^2 (inner) 
    from x=0 to x=1 in a 2.5D Isometric view.
    Includes buttons to toggle between Rotating Slices (Washers) and Rotating Functions (Solid).
    
    Parameters:
    - x_points: List of x-values to visualize as rotating washers.
    """
    # 1. Setup Data
    x_range = np.linspace(0, 1, 100)
    y_outer_curve = np.sqrt(x_range)
    y_inner_curve = x_range**2
    zeros = np.zeros_like(x_range)
    
    # 2. Initialize Figure
    fig = go.Figure()

    # --- TRACE 0: Axis of Rotation (X-Axis) ---
    fig.add_trace(go.Scatter3d(
        x=[-0.2, 1.2], y=[0, 0], z=[0, 0],
        mode='lines', line=dict(color='black', width=3),
        name='Axis of Rotation'
    ))

    # --- TRACE 1: Outer Curve (y = sqrt(x)) ---
    # Swapped y and z for 2.5D vertical look
    fig.add_trace(go.Scatter3d(
        x=x_range, y=zeros, z=y_outer_curve, 
        mode='lines', line=dict(color='blue', width=4),
        name='Outer: y=√x'
    ))

    # --- TRACE 2: Inner Curve (y = x^2) ---
    fig.add_trace(go.Scatter3d(
        x=x_range, y=zeros, z=y_inner_curve,
        mode='lines', line=dict(color='green', width=4),
        name='Inner: y=x²'
    ))

    # --- TRACE 3: Sample Points Markers (Start Position) ---
    xp = np.array(x_points)
    yp_out = np.sqrt(xp)
    yp_in = xp**2
    
    # Markers for both inner and outer radii on the curve
    fig.add_trace(go.Scatter3d(
        x=np.concatenate([xp, xp]), 
        y=np.zeros(2*len(xp)),
        z=np.concatenate([yp_out, yp_in]),
        mode='markers', marker=dict(size=5, color='red', symbol='circle'),
        name='Sample Points'
    ))
    
    # --- TRACES 4 to (4 + N-1): Sample Slices (Lines connecting inner/outer) ---
    # These represent the "representative rectangle" being rotated
    for i, val in enumerate(x_points):
        fig.add_trace(go.Scatter3d(
            x=[val, val], y=[0, 0], z=[yp_in[i], yp_out[i]],
            mode='lines', line=dict(color='red', width=4),
            showlegend=False, name=f'Slice x={val}'
        ))

    # --- TRACES for Washers (Placeholders to be animated) ---
    # Each washer needs: Outer Rim, Inner Rim, and the Face
    for x_val in x_points:
        # Outer Rim
        fig.add_trace(go.Scatter3d(x=[], y=[], z=[], mode='lines', line=dict(color='blue', width=2), showlegend=False))
        # Inner Rim
        fig.add_trace(go.Scatter3d(x=[], y=[], z=[], mode='lines', line=dict(color='green', width=2), showlegend=False))
        # Face (Mesh)
        fig.add_trace(go.Mesh3d(x=[], y=[], z=[], color='red', opacity=0.5, showscale=False, name=f'Washer x={x_val}'))

    # --- TRACES for Solid Surfaces (Placeholders) ---
    # Outer Surface
    fig.add_trace(go.Surface(
        x=[], y=[], z=[], 
        colorscale='Blues', opacity=0.3, showscale=False, 
        name='Outer Surface', visible=False
    ))
    # Inner Surface
    fig.add_trace(go.Surface(
        x=[], y=[], z=[], 
        colorscale='Greens', opacity=0.9, showscale=False, 
        name='Inner Surface', visible=False
    ))

    # --- CALCULATE VISIBILITY INDICES ---
    # Traces:
    # 0: Axis
    # 1: Outer Curve
    # 2: Inner Curve
    # 3: Markers
    # 4..(4+len(x)-1): Slice Lines
    # Then Washers (3 traces per point: Outer Rim, Inner Rim, Face)
    # Then 2 Surfaces (Outer, Inner)
    
    n_pts = len(x_points)
    idx_slices_start = 4
    idx_washers_start = 4 + n_pts
    n_washer_traces = 3 * n_pts
    idx_surfaces_start = idx_washers_start + n_washer_traces
    total_traces = idx_surfaces_start + 2
    
    # Toggle Arrays
    # Washer Mode: Show Axis(1), Curves(2), Markers(1), Slices(n), Washers(3n). Hide Surfaces(2).
    vis_washers = [True] * idx_surfaces_start + [False, False]
    
    # Solid Mode: Show Axis(1), Curves(2). Hide Markers, Slices, Washers. Show Surfaces(2).
    vis_solid = [True, True, True] + [False]*(1 + n_pts + n_washer_traces) + [True, True]

    # --- ANIMATION FRAMES ---
    frames = []
    steps = np.linspace(0, 2*np.pi, 36) # 36 frames
    
    for k, angle in enumerate(steps):
        frame_data = []
        is_last_frame = (k == len(steps) - 1)
        
        # 1. Static Traces
        frame_data.append(go.Scatter3d(x=[-0.2, 1.2], y=[0,0], z=[0,0]))
        frame_data.append(go.Scatter3d(x=x_range, y=zeros, z=y_outer_curve))
        frame_data.append(go.Scatter3d(x=x_range, y=zeros, z=y_inner_curve))
        
        # 2. Rotating Slices (Markers + Lines)
        # Calculate rotated positions
        # Standard Math Rotation: Y = r*cos, Z = r*sin
        # 2.5D Plot Swap: y_plot = Z_math, z_plot = Y_math
        
        # Markers
        # Outer
        y_math_out = yp_out * np.cos(angle)
        z_math_out = yp_out * np.sin(angle)
        # Inner
        y_math_in = yp_in * np.cos(angle)
        z_math_in = yp_in * np.sin(angle)
        
        frame_data.append(go.Scatter3d(
            x=np.concatenate([xp, xp]),
            y=np.concatenate([z_math_out, z_math_in]), # Swapped
            z=np.concatenate([y_math_out, y_math_in])  # Swapped
        ))
        
        # Slice Lines
        for i in range(n_pts):
            frame_data.append(go.Scatter3d(
                x=[xp[i], xp[i]],
                y=[z_math_in[i], z_math_out[i]], # Swapped
                z=[y_math_in[i], y_math_out[i]]  # Swapped
            ))
            
        # 3. Washers
        for i, x_val in enumerate(x_points):
            r_out = yp_out[i]
            r_in = yp_in[i]
            
            # Generate Arc Points
            num_arc_pts = max(2, int(30 * (angle/(2*np.pi))) + 2)
            t_slice = np.linspace(0, angle, num_arc_pts)
            
            # Math Coordinates
            Y_mo = r_out * np.cos(t_slice)
            Z_mo = r_out * np.sin(t_slice)
            Y_mi = r_in * np.cos(t_slice)
            Z_mi = r_in * np.sin(t_slice)
            
            # Plot Coordinates (Swapped)
            y_plot_out = Z_mo
            z_plot_out = Y_mo
            y_plot_in = Z_mi
            z_plot_in = Y_mi
            
            x_vec = np.full_like(t_slice, x_val)
            
            # Outer Rim
            frame_data.append(go.Scatter3d(x=x_vec, y=y_plot_out, z=z_plot_out))
            # Inner Rim
            frame_data.append(go.Scatter3d(x=x_vec, y=y_plot_in, z=z_plot_in))
            
            # Washer Face (Mesh)
            # Combine vertices: [OuterPts ..., InnerPts ...]
            x_mesh = np.concatenate([x_vec, x_vec])
            y_mesh = np.concatenate([y_plot_out, y_plot_in])
            z_mesh = np.concatenate([z_plot_out, z_plot_in])
            
            # Triangulate the strip between two arcs
            M = len(t_slice)
            if M >= 2:
                idx = np.arange(M - 1)
                # Triangles 1: (Out_i, Out_i+1, In_i)
                i1, j1, k1 = idx, idx + 1, idx + M
                # Triangles 2: (In_i, Out_i+1, In_i+1)
                i2, j2, k2 = idx + M, idx + 1, idx + M + 1
                
                i_idx = np.concatenate([i1, i2])
                j_idx = np.concatenate([j1, j2])
                k_idx = np.concatenate([k1, k2])
            else:
                i_idx, j_idx, k_idx = [], [], []
            
            frame_data.append(go.Mesh3d(
                x=x_mesh, y=y_mesh, z=z_mesh,
                i=i_idx, j=j_idx, k=k_idx,
                color='red', opacity=0.5
            ))

        # 4. Solid Surfaces
        if angle > 0:
            t_surf = np.linspace(0, angle, 30)
            X_S, T_S = np.meshgrid(x_range, t_surf)
            
            # Outer
            R_o = np.sqrt(X_S)
            Y_mo = R_o * np.cos(T_S)
            Z_mo = R_o * np.sin(T_S)
            frame_data.append(go.Surface(
                x=X_S, y=Z_mo, z=Y_mo, # Swapped
                colorscale='Blues', opacity=0.3
            ))
            
            # Inner
            R_i = X_S**2
            Y_mi = R_i * np.cos(T_S)
            Z_mi = R_i * np.sin(T_S)
            frame_data.append(go.Surface(
                x=X_S, y=Z_mi, z=Y_mi, # Swapped
                colorscale='Greens', opacity=0.9
            ))
        else:
            frame_data.append(go.Surface(x=[], y=[], z=[]))
            frame_data.append(go.Surface(x=[], y=[], z=[]))

        frames.append(go.Frame(data=frame_data, name=f'rot{k}'))

    fig.frames = frames

    # 4. Layout & Buttons
    updatemenus = [dict(
        type="buttons",
        direction="left",
        x=0.0, y=1.1,
        xanchor='left', yanchor='top',
        buttons=list([
            dict(
                label="Show Washers",
                method="update",
                args=[{"visible": vis_washers},
                    {"title": "Washer Method: Rotating Slices"}]
            ),
            dict(
                label="Show Solid",
                method="update",
                args=[{"visible": vis_solid},
                    {"title": "Washer Method: Full Volume"}]
            )
        ]),
    )]

    fig.update_layout(
        title="Washer Method",
        width=800,
        height=600,
        updatemenus=updatemenus,
        scene=dict(
            xaxis=dict(title='X (Axis)', range=[0, 1.2], showgrid=True),
            # Labels removed for Y and Z
            yaxis=dict(title='', range=[-1.2, 1.2], showgrid=True, showticklabels=False),
            zaxis=dict(title='', range=[-1.2, 1.2], showgrid=True, showticklabels=False),
            aspectmode='cube',
            
            # --- 2.5D Isometric Settings ---
            camera=dict(
                projection=dict(type="orthographic"),
                eye=dict(x=1.5, y=1.5, z=1.5),
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

# Run the 2.5D Washer visualization
visualize_washer_method_25d(x_points=[0.2, 0.5, 0.8])