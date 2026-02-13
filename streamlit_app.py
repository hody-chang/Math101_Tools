import streamlit as st
import numpy as np
import plotly.graph_objects as go

# --- PAGE CONFIGURATION ---
st.set_page_config(layout="wide", page_title="Calculus Visualizer")

st.title("Calculus: Volume of Revolution Visualizer")
st.markdown("""
This tool helps visualize how 2D areas are rotated around an axis to create 3D volumes.
Select a method from the sidebar and define your own functions.
""")

# --- HELPER: SAFE FUNCTION EVALUATION ---
def safe_evaluate(func_str, var_arr):
    """
    Evaluates a mathematical string safely using numpy.
    Handles 'x' or 'y' as the variable and common math functions.
    """
    # Allow students to use ^ for power instead of **
    func_str = func_str.replace("^", "**")
    
    allowed_locals = {
        "x": var_arr, "y": var_arr,
        "sin": np.sin, "cos": np.cos, "tan": np.tan,
        "sqrt": np.sqrt, "exp": np.exp, "log": np.log, "ln": np.log,
        "pi": np.pi, "e": np.e, "abs": np.abs,
        "power": np.power
    }
    
    try:
        return eval(func_str, {"__builtins__": {}}, allowed_locals)
    except Exception as e:
        return None

# --- SIDEBAR CONTROLS ---
st.sidebar.header("Settings")
method = st.sidebar.radio("Choose Method", ["Disk Method", "Washer Method"])

# --- VISUALIZATION FUNCTIONS ---

def get_disk_method_fig(func_str, t_min, t_max, t_points_input, axis_type, axis_value):
    """
    Disk method visualization.
    - axis_type: "y = c" or "x = c"
    - axis_value: the constant c
    For y=c: rotate y=f(x) around horizontal line y=c. Variable is x.
    For x=c: rotate x=g(y) around vertical line x=c. Variable is y.
    """
    is_horizontal = (axis_type == "y = c")
    
    # t is the integration variable (x for y=c, y for x=c)
    t_range = np.linspace(t_min, t_max, 100)
    f_vals = safe_evaluate(func_str, t_range)
    if f_vals is None:
        return go.Figure().add_annotation(text="Invalid Function Syntax", showarrow=False)
    # Ensure f_vals is an array
    if np.isscalar(f_vals):
        f_vals = np.full_like(t_range, f_vals)
    
    padding = (t_max - t_min) * 0.1
    fig = go.Figure()
    
    if is_horizontal:
        # --- Rotation around y = c (horizontal line) ---
        # 3D coords: X-axis = x, Y/Z plane = rotation plane
        # Curve sits at (x, 0, f(x)) in 3D. Axis line sits at (x, 0, c).
        # Radius at each x: R(x) = |f(x) - c|
        
        # Axis of rotation line
        fig.add_trace(go.Scatter3d(
            x=[t_min-padding, t_max+padding], y=[0, 0], z=[axis_value, axis_value],
            mode='lines', line=dict(color='black', width=3, dash='dash'), name=f'Axis y={axis_value}'))
        # Curve
        fig.add_trace(go.Scatter3d(
            x=t_range, y=np.zeros_like(t_range), z=f_vals,
            mode='lines', line=dict(color='blue', width=5), name=f'y={func_str}'))
        
        # Radius at each point
        R_curve = f_vals - axis_value  # Signed: positive means above axis, negative means below
        
        # Points
        tp = np.array(t_points_input)
        tp = tp[(tp >= t_min) & (tp <= t_max)]
        
        if len(tp) > 0:
            fp = safe_evaluate(func_str, tp)
            if np.isscalar(fp):
                fp = np.full_like(tp, fp)
            Rp = fp - axis_value  # Signed radius
            fig.add_trace(go.Scatter3d(x=tp, y=np.zeros_like(tp), z=fp,
                                       mode='markers', marker=dict(size=6, color='red'), name='Points'))
        else:
            fp, Rp = np.array([]), np.array([])
        
        # Placeholder Traces for Disks (Rim + Face per point)
        for _ in tp:
            fig.add_trace(go.Scatter3d(x=[], y=[], z=[], mode='lines', line=dict(color='red', width=3), showlegend=False))
            fig.add_trace(go.Mesh3d(x=[], y=[], z=[], color='red', opacity=0.5, showscale=False))

        # Placeholder for Solid Surface + Caps
        fig.add_trace(go.Surface(x=[], y=[], z=[], colorscale='Viridis', opacity=0.8, showscale=False, name='Solid', visible=False))
        fig.add_trace(go.Surface(x=[], y=[], z=[], colorscale='Viridis', opacity=0.8, showscale=False, name='Cap Start', visible=False))
        fig.add_trace(go.Surface(x=[], y=[], z=[], colorscale='Viridis', opacity=0.8, showscale=False, name='Cap End', visible=False))

        # Visibility Logic
        num_disk_traces = 2 * len(tp)
        vis_disks = [True] * (3 + num_disk_traces) + [False]*3
        vis_solid = [True, True] + [False]*(1 + num_disk_traces) + [True, True, True]

        # Animation Frames
        frames = []
        steps = np.linspace(0, 2*np.pi, 36)
        
        for k, angle in enumerate(steps):
            frame_data = []
            is_last = (k == len(steps) - 1)
            
            # Axis line (static)
            frame_data.append(go.Scatter3d(x=[t_min-padding, t_max+padding], y=[0, 0], z=[axis_value, axis_value]))
            # Curve (static)
            frame_data.append(go.Scatter3d(x=t_range, y=np.zeros_like(t_range), z=f_vals))
            
            # Rotating Points
            if len(tp) > 0:
                y_rot = axis_value + Rp * np.cos(angle)
                z_rot = Rp * np.sin(angle)
                frame_data.append(go.Scatter3d(x=tp, y=z_rot, z=y_rot))
            else:
                frame_data.append(go.Scatter3d(x=[], y=[], z=[]))
            
            # Disks
            for i, x_val in enumerate(tp):
                r = Rp[i]
                num_pts = max(2, int(30 * (angle/(2*np.pi))) + 2)
                t_slice = np.linspace(0, angle, num_pts)
                
                cy = axis_value + r * np.cos(t_slice)
                cz = r * np.sin(t_slice)
                cx = np.full_like(t_slice, x_val)
                
                # Rim
                frame_data.append(go.Scatter3d(x=cx, y=cz, z=cy))
                
                # Face (fan from center)
                x_m = np.concatenate(([x_val], cx))
                y_m = np.concatenate(([0], cz))
                z_m = np.concatenate(([axis_value], cy))
                
                M = len(t_slice)
                if M >= 2:
                    i_idx = [0]*(M-1)
                    j_idx = list(range(1, M))
                    k_idx = list(range(2, M+1))
                else:
                    i_idx, j_idx, k_idx = [], [], []
                
                op = 1.0 if is_last else 0.5
                frame_data.append(go.Mesh3d(x=x_m, y=y_m, z=z_m, i=i_idx, j=j_idx, k=k_idx, color='red', opacity=op))
            
            # Solid Surface + Caps
            if angle > 0:
                op_solid = 1.0 if is_last else 0.8
                t_slice = np.linspace(0, angle, 30)
                X_S, T_S = np.meshgrid(t_range, t_slice)
                R_S = np.tile(R_curve, (len(t_slice), 1))
                
                # Main Body: centered on axis_value
                Y_body = R_S * np.sin(T_S)
                Z_body = axis_value + R_S * np.cos(T_S)
                frame_data.append(go.Surface(x=X_S, y=Y_body, z=Z_body, colorscale='Viridis', opacity=op_solid))

                # Cap Start (x_min)
                r_start = np.linspace(0, abs(R_curve[0]), 2)
                R_c1, T_c1 = np.meshgrid(r_start, t_slice)
                frame_data.append(go.Surface(
                    x=np.full_like(R_c1, t_min), y=R_c1*np.sin(T_c1), z=axis_value + R_c1*np.cos(T_c1),
                    colorscale='Viridis', opacity=op_solid, showscale=False))

                # Cap End (x_max)
                r_end = np.linspace(0, abs(R_curve[-1]), 2)
                R_c2, T_c2 = np.meshgrid(r_end, t_slice)
                frame_data.append(go.Surface(
                    x=np.full_like(R_c2, t_max), y=R_c2*np.sin(T_c2), z=axis_value + R_c2*np.cos(T_c2),
                    colorscale='Viridis', opacity=op_solid, showscale=False))
            else:
                frame_data.append(go.Surface(x=[], y=[], z=[]))
                frame_data.append(go.Surface(x=[], y=[], z=[]))
                frame_data.append(go.Surface(x=[], y=[], z=[]))
            
            frames.append(go.Frame(data=frame_data, name=f'rot{k}'))
        
        fig.frames = frames
        
        # Calculate bounds for consistent view
        abs_R_curve = np.abs(R_curve)
        r_max = np.max(abs_R_curve) if len(abs_R_curve) > 0 else 1.0
        yz_lo = axis_value - r_max * 1.3
        yz_hi = axis_value + r_max * 1.3
        # Also make sure the curve is visible
        yz_lo = min(yz_lo, np.min(f_vals) - 0.2)
        yz_hi = max(yz_hi, np.max(f_vals) + 0.2)
        y_lim = [-r_max*1.3, r_max*1.3]  # y-axis (depth) centered on 0
        z_lim = [yz_lo, yz_hi]

        fig.update_layout(
            title=f"Disk Method: y = {func_str}, rotated about y = {axis_value}",
            height=600,
            template="plotly_white",
            margin=dict(t=100, b=0, l=0, r=0),
            uirevision='constant',
            updatemenus=[dict(
                type="buttons", direction="left", x=0, y=1.1,
                buttons=[
                    dict(label="Show Disks", method="update", args=[{"visible": vis_disks}]),
                    dict(label="Show Solid", method="update", args=[{"visible": vis_solid}])
                ]
            )],
            scene=dict(
                dragmode="orbit",
                xaxis=dict(title='x', range=[t_min, t_max]),
                yaxis=dict(title='', range=y_lim, showticklabels=False),
                zaxis=dict(title='y', range=z_lim),
                aspectmode='cube',
                camera=dict(projection=dict(type="orthographic"), eye=dict(x=0.3, y=-1.8, z=0.3), up=dict(x=0, y=0, z=1))
            ),
            sliders=[dict(
                steps=[dict(method='animate', args=[[f'rot{k}'], dict(mode='immediate', frame=dict(duration=20, redraw=True))], label='') for k in range(len(steps))],
                currentvalue=dict(visible=False)
            )]
        )
    else:
        # --- Rotation around x = c (vertical line) ---
        # Variable is y. Function gives x = g(y).
        # 3D coords: Y-axis = y (integration direction), X/Z plane = rotation plane
        # Curve sits at (g(y), y, 0). Axis at (c, y, 0).
        # Radius at each y: R(y) = |g(y) - c|
        
        y_range = t_range  # t is y here
        g_vals = f_vals     # g(y) values
        
        R_curve = g_vals - axis_value  # Signed radius
        
        # Axis of rotation line
        fig.add_trace(go.Scatter3d(
            x=[axis_value, axis_value], y=[t_min-padding, t_max+padding], z=[0, 0],
            mode='lines', line=dict(color='black', width=3, dash='dash'), name=f'Axis x={axis_value}'))
        # Curve
        fig.add_trace(go.Scatter3d(
            x=g_vals, y=y_range, z=np.zeros_like(y_range),
            mode='lines', line=dict(color='blue', width=5), name=f'x={func_str}'))
        
        # Points
        tp = np.array(t_points_input)
        tp = tp[(tp >= t_min) & (tp <= t_max)]
        
        if len(tp) > 0:
            gp = safe_evaluate(func_str, tp)
            if np.isscalar(gp):
                gp = np.full_like(tp, gp)
            Rp = gp - axis_value  # Signed radius
            fig.add_trace(go.Scatter3d(x=gp, y=tp, z=np.zeros_like(tp),
                                       mode='markers', marker=dict(size=6, color='red'), name='Points'))
        else:
            gp, Rp = np.array([]), np.array([])
        
        # Placeholder Traces for Disks
        for _ in tp:
            fig.add_trace(go.Scatter3d(x=[], y=[], z=[], mode='lines', line=dict(color='red', width=3), showlegend=False))
            fig.add_trace(go.Mesh3d(x=[], y=[], z=[], color='red', opacity=0.5, showscale=False))

        # Placeholders for Solid + Caps
        fig.add_trace(go.Surface(x=[], y=[], z=[], colorscale='Viridis', opacity=0.8, showscale=False, name='Solid', visible=False))
        fig.add_trace(go.Surface(x=[], y=[], z=[], colorscale='Viridis', opacity=0.8, showscale=False, name='Cap Start', visible=False))
        fig.add_trace(go.Surface(x=[], y=[], z=[], colorscale='Viridis', opacity=0.8, showscale=False, name='Cap End', visible=False))

        num_disk_traces = 2 * len(tp)
        vis_disks = [True] * (3 + num_disk_traces) + [False]*3
        vis_solid = [True, True] + [False]*(1 + num_disk_traces) + [True, True, True]

        frames = []
        steps = np.linspace(0, 2*np.pi, 36)
        
        for k, angle in enumerate(steps):
            frame_data = []
            is_last = (k == len(steps) - 1)
            
            # Axis (static)
            frame_data.append(go.Scatter3d(x=[axis_value, axis_value], y=[t_min-padding, t_max+padding], z=[0, 0]))
            # Curve (static)
            frame_data.append(go.Scatter3d(x=g_vals, y=y_range, z=np.zeros_like(y_range)))
            
            # Rotating Points
            if len(tp) > 0:
                x_rot = axis_value + Rp * np.cos(angle)
                z_rot = Rp * np.sin(angle)
                frame_data.append(go.Scatter3d(x=x_rot, y=tp, z=z_rot))
            else:
                frame_data.append(go.Scatter3d(x=[], y=[], z=[]))
            
            # Disks
            for i, y_val in enumerate(tp):
                r = Rp[i]
                num_pts = max(2, int(30 * (angle/(2*np.pi))) + 2)
                t_slice = np.linspace(0, angle, num_pts)
                
                cx = axis_value + r * np.cos(t_slice)
                cz = r * np.sin(t_slice)
                cy = np.full_like(t_slice, y_val)
                
                # Rim
                frame_data.append(go.Scatter3d(x=cx, y=cy, z=cz))
                
                # Face
                x_m = np.concatenate(([axis_value], cx))
                y_m = np.concatenate(([y_val], cy))
                z_m = np.concatenate(([0], cz))
                
                M = len(t_slice)
                if M >= 2:
                    i_idx = [0]*(M-1)
                    j_idx = list(range(1, M))
                    k_idx = list(range(2, M+1))
                else:
                    i_idx, j_idx, k_idx = [], [], []
                
                op = 1.0 if is_last else 0.5
                frame_data.append(go.Mesh3d(x=x_m, y=y_m, z=z_m, i=i_idx, j=j_idx, k=k_idx, color='red', opacity=op))
            
            # Solid Surface + Caps
            if angle > 0:
                op_solid = 1.0 if is_last else 0.8
                t_slice = np.linspace(0, angle, 30)
                Y_S, T_S = np.meshgrid(y_range, t_slice)
                R_S = np.tile(R_curve, (len(t_slice), 1))
                
                X_body = axis_value + R_S * np.cos(T_S)
                Z_body = R_S * np.sin(T_S)
                frame_data.append(go.Surface(x=X_body, y=Y_S, z=Z_body, colorscale='Viridis', opacity=op_solid))

                # Cap Start (y_min)
                r_start = np.linspace(0, abs(R_curve[0]), 2)
                R_c1, T_c1 = np.meshgrid(r_start, t_slice)
                frame_data.append(go.Surface(
                    x=axis_value + R_c1*np.cos(T_c1), y=np.full_like(R_c1, t_min), z=R_c1*np.sin(T_c1),
                    colorscale='Viridis', opacity=op_solid, showscale=False))

                # Cap End (y_max)
                r_end = np.linspace(0, abs(R_curve[-1]), 2)
                R_c2, T_c2 = np.meshgrid(r_end, t_slice)
                frame_data.append(go.Surface(
                    x=axis_value + R_c2*np.cos(T_c2), y=np.full_like(R_c2, t_max), z=R_c2*np.sin(T_c2),
                    colorscale='Viridis', opacity=op_solid, showscale=False))
            else:
                frame_data.append(go.Surface(x=[], y=[], z=[]))
                frame_data.append(go.Surface(x=[], y=[], z=[]))
                frame_data.append(go.Surface(x=[], y=[], z=[]))
            
            frames.append(go.Frame(data=frame_data, name=f'rot{k}'))
        
        fig.frames = frames
        
        abs_R_curve = np.abs(R_curve)
        r_max = np.max(abs_R_curve) if len(abs_R_curve) > 0 else 1.0
        x_lo = min(axis_value - r_max*1.3, np.min(g_vals) - 0.2)
        x_hi = max(axis_value + r_max*1.3, np.max(g_vals) + 0.2)
        z_lim = [-r_max*1.3, r_max*1.3]

        fig.update_layout(
            title=f"Disk Method: x = {func_str}, rotated about x = {axis_value}",
            height=600,
            template="plotly_white",
            margin=dict(t=100, b=0, l=0, r=0),
            uirevision='constant',
            updatemenus=[dict(
                type="buttons", direction="left", x=0, y=1.1,
                buttons=[
                    dict(label="Show Disks", method="update", args=[{"visible": vis_disks}]),
                    dict(label="Show Solid", method="update", args=[{"visible": vis_solid}])
                ]
            )],
            scene=dict(
                dragmode="orbit",
                xaxis=dict(title='x', range=[x_lo, x_hi]),
                yaxis=dict(title='y', range=[t_min, t_max]),
                zaxis=dict(title='', range=z_lim, showticklabels=False),
                aspectmode='cube',
                camera=dict(projection=dict(type="orthographic"), eye=dict(x=0.3, y=0.3, z=1.8), up=dict(x=0, y=1, z=0))
            ),
            sliders=[dict(
                steps=[dict(method='animate', args=[[f'rot{k}'], dict(mode='immediate', frame=dict(duration=20, redraw=True))], label='') for k in range(len(steps))],
                currentvalue=dict(visible=False)
            )]
        )

    return fig


def get_washer_method_fig(func_out_str, func_in_str, t_min, t_max, t_points_input, axis_type, axis_value):
    """
    Washer method visualization.
    - axis_type: "y = c" or "x = c"
    - axis_value: the constant c
    """
    is_horizontal = (axis_type == "y = c")
    
    t_range = np.linspace(t_min, t_max, 100)
    
    f_outer = safe_evaluate(func_out_str, t_range)
    f_inner = safe_evaluate(func_in_str, t_range)
    
    if f_outer is None or f_inner is None:
        return go.Figure().add_annotation(text="Invalid Function Syntax", showarrow=False)
    if np.isscalar(f_outer):
        f_outer = np.full_like(t_range, f_outer)
    if np.isscalar(f_inner):
        f_inner = np.full_like(t_range, f_inner)
    
    padding = (t_max - t_min) * 0.1
    fig = go.Figure()
    
    if is_horizontal:
        # --- Rotation around y = c ---
        R_outer = f_outer - axis_value  # Signed radius
        R_inner = f_inner - axis_value  # Signed radius
        
        # Axis line
        fig.add_trace(go.Scatter3d(
            x=[t_min-padding, t_max+padding], y=[0, 0], z=[axis_value, axis_value],
            mode='lines', line=dict(color='black', width=3, dash='dash'), name=f'Axis y={axis_value}'))
        # Outer curve
        fig.add_trace(go.Scatter3d(
            x=t_range, y=np.zeros_like(t_range), z=f_outer,
            mode='lines', line=dict(color='blue', width=4), name=f'Outer: {func_out_str}'))
        # Inner curve
        fig.add_trace(go.Scatter3d(
            x=t_range, y=np.zeros_like(t_range), z=f_inner,
            mode='lines', line=dict(color='green', width=4), name=f'Inner: {func_in_str}'))

        # Sample Points & Slices
        tp = np.array(t_points_input)
        tp = tp[(tp >= t_min) & (tp <= t_max)]
        
        if len(tp) > 0:
            fp_out = safe_evaluate(func_out_str, tp)
            fp_in = safe_evaluate(func_in_str, tp)
            if np.isscalar(fp_out):
                fp_out = np.full_like(tp, fp_out)
            if np.isscalar(fp_in):
                fp_in = np.full_like(tp, fp_in)
            Rp_out = fp_out - axis_value  # Signed
            Rp_in = fp_in - axis_value  # Signed
            
            fig.add_trace(go.Scatter3d(
                x=np.concatenate([tp, tp]), y=np.zeros(2*len(tp)),
                z=np.concatenate([fp_out, fp_in]),
                mode='markers', marker=dict(color='red', size=5), name='Points'))
            
            for i, val in enumerate(tp):
                fig.add_trace(go.Scatter3d(
                    x=[val, val], y=[0, 0], z=[fp_in[i], fp_out[i]],
                    mode='lines', line=dict(color='red', width=4), showlegend=False))
        else:
            fp_out, fp_in = np.array([]), np.array([])
            Rp_out, Rp_in = np.array([]), np.array([])
        
        n_pts = len(tp)
        
        # Placeholders for Washers (3 traces per washer: OutRim, InRim, Face)
        for _ in tp:
            fig.add_trace(go.Scatter3d(x=[], y=[], z=[], mode='lines', line=dict(color='blue', width=2), showlegend=False))
            fig.add_trace(go.Scatter3d(x=[], y=[], z=[], mode='lines', line=dict(color='green', width=2), showlegend=False))
            fig.add_trace(go.Mesh3d(x=[], y=[], z=[], color='red', opacity=0.5, showscale=False))

        # Placeholders for Surfaces (Outer, Inner) + Caps
        fig.add_trace(go.Surface(x=[], y=[], z=[], colorscale='Blues', opacity=0.3, showscale=False, visible=False))
        fig.add_trace(go.Surface(x=[], y=[], z=[], colorscale='Greens', opacity=0.9, showscale=False, visible=False))
        fig.add_trace(go.Surface(x=[], y=[], z=[], colorscale='Greens', opacity=0.9, showscale=False, visible=False))
        fig.add_trace(go.Surface(x=[], y=[], z=[], colorscale='Greens', opacity=0.9, showscale=False, visible=False))

        # Visibility logic
        idx_solids_start = 4 + n_pts + 3*n_pts
        vis_washers = [True] * idx_solids_start + [False]*4
        vis_solid = [True, True, True] + [False]*(1 + n_pts + 3*n_pts) + [True, True, True, True]

        # Animation
        frames = []
        steps = np.linspace(0, 2*np.pi, 36)
        
        for k, angle in enumerate(steps):
            frame_data = []
            is_last = (k == len(steps) - 1)
            
            # Static traces
            frame_data.append(go.Scatter3d(x=[t_min-padding, t_max+padding], y=[0, 0], z=[axis_value, axis_value]))
            frame_data.append(go.Scatter3d(x=t_range, y=np.zeros_like(t_range), z=f_outer))
            frame_data.append(go.Scatter3d(x=t_range, y=np.zeros_like(t_range), z=f_inner))
            
            # Rotating Markers
            if len(tp) > 0:
                y_mo = axis_value + Rp_out * np.cos(angle)
                z_mo = Rp_out * np.sin(angle)
                y_mi = axis_value + Rp_in * np.cos(angle)
                z_mi = Rp_in * np.sin(angle)
                
                frame_data.append(go.Scatter3d(
                    x=np.concatenate([tp, tp]),
                    y=np.concatenate([z_mo, z_mi]),
                    z=np.concatenate([y_mo, y_mi])))
                
                for i in range(n_pts):
                    frame_data.append(go.Scatter3d(x=[tp[i], tp[i]], y=[z_mi[i], z_mo[i]], z=[y_mi[i], y_mo[i]]))

                for i, val in enumerate(tp):
                    num_arc = max(2, int(30 * (angle/(2*np.pi))) + 2)
                    t = np.linspace(0, angle, num_arc)
                    x_vec = np.full_like(t, val)
                    
                    yo = axis_value + Rp_out[i] * np.cos(t)
                    zo = Rp_out[i] * np.sin(t)
                    yi = axis_value + Rp_in[i] * np.cos(t)
                    zi = Rp_in[i] * np.sin(t)
                    
                    frame_data.append(go.Scatter3d(x=x_vec, y=zo, z=yo))
                    frame_data.append(go.Scatter3d(x=x_vec, y=zi, z=yi))
                    
                    x_m = np.concatenate([x_vec, x_vec])
                    y_m = np.concatenate([zo, zi])
                    z_m = np.concatenate([yo, yi])
                    
                    M = len(t)
                    if M >= 2:
                        idx = np.arange(M-1)
                        i_idx = np.concatenate([idx, idx+M])
                        j_idx = np.concatenate([idx+1, idx+1])
                        k_idx = np.concatenate([idx+M, idx+M+1])
                    else:
                        i_idx, j_idx, k_idx = [], [], []
                    
                    op = 1.0 if is_last else 0.5
                    frame_data.append(go.Mesh3d(x=x_m, y=y_m, z=z_m, i=i_idx, j=j_idx, k=k_idx, color='red', opacity=op))
            else:
                frame_data.append(go.Scatter3d(x=[], y=[], z=[]))
            
            # Solid + Caps
            if angle > 0:
                t_surf = np.linspace(0, angle, 30)
                X_S, T_S = np.meshgrid(t_range, t_surf)
                
                cs_out = [[0, 'rgb(255, 150, 150)'], [1, 'rgb(255, 150, 150)']] if is_last else 'Blues'
                op_out = 1.0 if is_last else 0.3
                cs_in = [[0, 'rgb(255, 150, 150)'], [1, 'rgb(255, 150, 150)']] if is_last else 'Greens'
                op_in = 1.0 if is_last else 0.9

                R_o = np.tile(R_outer, (len(t_surf), 1))
                Y_body_o = R_o * np.sin(T_S)
                Z_body_o = axis_value + R_o * np.cos(T_S)
                frame_data.append(go.Surface(x=X_S, y=Y_body_o, z=Z_body_o, colorscale=cs_out, opacity=op_out, showscale=False))
                
                R_i = np.tile(R_inner, (len(t_surf), 1))
                Y_body_i = R_i * np.sin(T_S)
                Z_body_i = axis_value + R_i * np.cos(T_S)
                frame_data.append(go.Surface(x=X_S, y=Y_body_i, z=Z_body_i, colorscale=cs_in, opacity=op_in, showscale=False))

                # Cap Start
                r_cap_start = np.linspace(R_inner[0], R_outer[0], 2)
                R_c1, T_c1 = np.meshgrid(r_cap_start, t_surf)
                frame_data.append(go.Surface(
                    x=np.full_like(R_c1, t_min),
                    y=R_c1*np.sin(T_c1),
                    z=axis_value + R_c1*np.cos(T_c1),
                    colorscale=cs_in, opacity=op_in, showscale=False))
                
                # Cap End
                r_cap_end = np.linspace(R_inner[-1], R_outer[-1], 2)
                R_c2, T_c2 = np.meshgrid(r_cap_end, t_surf)
                frame_data.append(go.Surface(
                    x=np.full_like(R_c2, t_max),
                    y=R_c2*np.sin(T_c2),
                    z=axis_value + R_c2*np.cos(T_c2),
                    colorscale=cs_in, opacity=op_in, showscale=False))
            else:
                frame_data.append(go.Surface(x=[], y=[], z=[]))
                frame_data.append(go.Surface(x=[], y=[], z=[]))
                frame_data.append(go.Surface(x=[], y=[], z=[]))
                frame_data.append(go.Surface(x=[], y=[], z=[]))
            
            frames.append(go.Frame(data=frame_data, name=f'rot{k}'))
        
        fig.frames = frames
        
        max_R = max(np.max(np.abs(R_outer)), np.max(np.abs(R_inner))) if len(R_outer) > 0 else 1.0
        y_lim = [-max_R*1.3, max_R*1.3]
        z_lo = min(axis_value - max_R*1.3, min(np.min(f_outer), np.min(f_inner)) - 0.2)
        z_hi = max(axis_value + max_R*1.3, max(np.max(f_outer), np.max(f_inner)) + 0.2)

        fig.update_layout(
            title=f"Washer Method: {func_out_str} vs {func_in_str}, rotated about y = {axis_value}",
            height=600,
            template="plotly_white",
            margin=dict(t=100, b=0, l=0, r=0),
            uirevision='constant',
            updatemenus=[dict(
                type="buttons", direction="left", x=0, y=1.1,
                buttons=[
                    dict(label="Show Washers", method="update", args=[{"visible": vis_washers}]),
                    dict(label="Show Solid", method="update", args=[{"visible": vis_solid}])
                ]
            )],
            scene=dict(
                dragmode="orbit",
                xaxis=dict(title='x', range=[t_min, t_max]),
                yaxis=dict(title='', range=y_lim, showticklabels=False),
                zaxis=dict(title='y', range=[z_lo, z_hi]),
                aspectmode='cube',
                camera=dict(projection=dict(type="orthographic"), eye=dict(x=0.3, y=-1.8, z=0.3), up=dict(x=0, y=0, z=1))
            ),
            sliders=[dict(
                steps=[dict(method='animate', args=[[f'rot{k}'], dict(mode='immediate', frame=dict(duration=20, redraw=True))], label='') for k in range(len(steps))],
                currentvalue=dict(visible=False)
            )]
        )

    else:
        # --- Rotation around x = c (vertical line) ---
        R_outer = f_outer - axis_value  # Signed
        R_inner = f_inner - axis_value  # Signed
        y_range = t_range
        
        # Axis line
        fig.add_trace(go.Scatter3d(
            x=[axis_value, axis_value], y=[t_min-padding, t_max+padding], z=[0, 0],
            mode='lines', line=dict(color='black', width=3, dash='dash'), name=f'Axis x={axis_value}'))
        # Outer curve
        fig.add_trace(go.Scatter3d(
            x=f_outer, y=y_range, z=np.zeros_like(y_range),
            mode='lines', line=dict(color='blue', width=4), name=f'Outer: {func_out_str}'))
        # Inner curve
        fig.add_trace(go.Scatter3d(
            x=f_inner, y=y_range, z=np.zeros_like(y_range),
            mode='lines', line=dict(color='green', width=4), name=f'Inner: {func_in_str}'))

        # Sample Points
        tp = np.array(t_points_input)
        tp = tp[(tp >= t_min) & (tp <= t_max)]
        
        if len(tp) > 0:
            gp_out = safe_evaluate(func_out_str, tp)
            gp_in = safe_evaluate(func_in_str, tp)
            if np.isscalar(gp_out):
                gp_out = np.full_like(tp, gp_out)
            if np.isscalar(gp_in):
                gp_in = np.full_like(tp, gp_in)
            Rp_out = gp_out - axis_value  # Signed
            Rp_in = gp_in - axis_value  # Signed
            
            fig.add_trace(go.Scatter3d(
                x=np.concatenate([gp_out, gp_in]), y=np.concatenate([tp, tp]),
                z=np.zeros(2*len(tp)),
                mode='markers', marker=dict(color='red', size=5), name='Points'))
            
            for i, val in enumerate(tp):
                fig.add_trace(go.Scatter3d(
                    x=[gp_in[i], gp_out[i]], y=[val, val], z=[0, 0],
                    mode='lines', line=dict(color='red', width=4), showlegend=False))
        else:
            gp_out, gp_in = np.array([]), np.array([])
            Rp_out, Rp_in = np.array([]), np.array([])
        
        n_pts = len(tp)
        
        # Placeholders for Washers
        for _ in tp:
            fig.add_trace(go.Scatter3d(x=[], y=[], z=[], mode='lines', line=dict(color='blue', width=2), showlegend=False))
            fig.add_trace(go.Scatter3d(x=[], y=[], z=[], mode='lines', line=dict(color='green', width=2), showlegend=False))
            fig.add_trace(go.Mesh3d(x=[], y=[], z=[], color='red', opacity=0.5, showscale=False))

        # Placeholders for Surfaces + Caps
        fig.add_trace(go.Surface(x=[], y=[], z=[], colorscale='Blues', opacity=0.3, showscale=False, visible=False))
        fig.add_trace(go.Surface(x=[], y=[], z=[], colorscale='Greens', opacity=0.9, showscale=False, visible=False))
        fig.add_trace(go.Surface(x=[], y=[], z=[], colorscale='Greens', opacity=0.9, showscale=False, visible=False))
        fig.add_trace(go.Surface(x=[], y=[], z=[], colorscale='Greens', opacity=0.9, showscale=False, visible=False))

        idx_solids_start = 4 + n_pts + 3*n_pts
        vis_washers = [True] * idx_solids_start + [False]*4
        vis_solid = [True, True, True] + [False]*(1 + n_pts + 3*n_pts) + [True, True, True, True]

        frames = []
        steps = np.linspace(0, 2*np.pi, 36)
        
        for k, angle in enumerate(steps):
            frame_data = []
            is_last = (k == len(steps) - 1)
            
            # Static
            frame_data.append(go.Scatter3d(x=[axis_value, axis_value], y=[t_min-padding, t_max+padding], z=[0, 0]))
            frame_data.append(go.Scatter3d(x=f_outer, y=y_range, z=np.zeros_like(y_range)))
            frame_data.append(go.Scatter3d(x=f_inner, y=y_range, z=np.zeros_like(y_range)))
            
            if len(tp) > 0:
                x_mo = axis_value + Rp_out * np.cos(angle)
                z_mo = Rp_out * np.sin(angle)
                x_mi = axis_value + Rp_in * np.cos(angle)
                z_mi = Rp_in * np.sin(angle)
                
                frame_data.append(go.Scatter3d(
                    x=np.concatenate([x_mo, x_mi]),
                    y=np.concatenate([tp, tp]),
                    z=np.concatenate([z_mo, z_mi])))
                
                for i in range(n_pts):
                    frame_data.append(go.Scatter3d(x=[x_mi[i], x_mo[i]], y=[tp[i], tp[i]], z=[z_mi[i], z_mo[i]]))

                for i, val in enumerate(tp):
                    num_arc = max(2, int(30 * (angle/(2*np.pi))) + 2)
                    t = np.linspace(0, angle, num_arc)
                    y_vec = np.full_like(t, val)
                    
                    xo = axis_value + Rp_out[i] * np.cos(t)
                    zo = Rp_out[i] * np.sin(t)
                    xi = axis_value + Rp_in[i] * np.cos(t)
                    zi = Rp_in[i] * np.sin(t)
                    
                    frame_data.append(go.Scatter3d(x=xo, y=y_vec, z=zo))
                    frame_data.append(go.Scatter3d(x=xi, y=y_vec, z=zi))
                    
                    x_m = np.concatenate([xo, xi])
                    y_m = np.concatenate([y_vec, y_vec])
                    z_m = np.concatenate([zo, zi])
                    
                    M = len(t)
                    if M >= 2:
                        idx = np.arange(M-1)
                        i_idx = np.concatenate([idx, idx+M])
                        j_idx = np.concatenate([idx+1, idx+1])
                        k_idx = np.concatenate([idx+M, idx+M+1])
                    else:
                        i_idx, j_idx, k_idx = [], [], []
                    
                    op = 1.0 if is_last else 0.5
                    frame_data.append(go.Mesh3d(x=x_m, y=y_m, z=z_m, i=i_idx, j=j_idx, k=k_idx, color='red', opacity=op))
            else:
                frame_data.append(go.Scatter3d(x=[], y=[], z=[]))
            
            # Solid + Caps
            if angle > 0:
                t_surf = np.linspace(0, angle, 30)
                Y_S, T_S = np.meshgrid(y_range, t_surf)
                
                cs_out = [[0, 'rgb(255, 150, 150)'], [1, 'rgb(255, 150, 150)']] if is_last else 'Blues'
                op_out = 1.0 if is_last else 0.3
                cs_in = [[0, 'rgb(255, 150, 150)'], [1, 'rgb(255, 150, 150)']] if is_last else 'Greens'
                op_in = 1.0 if is_last else 0.9

                R_o = np.tile(R_outer, (len(t_surf), 1))
                X_body_o = axis_value + R_o * np.cos(T_S)
                Z_body_o = R_o * np.sin(T_S)
                frame_data.append(go.Surface(x=X_body_o, y=Y_S, z=Z_body_o, colorscale=cs_out, opacity=op_out, showscale=False))
                
                R_i = np.tile(R_inner, (len(t_surf), 1))
                X_body_i = axis_value + R_i * np.cos(T_S)
                Z_body_i = R_i * np.sin(T_S)
                frame_data.append(go.Surface(x=X_body_i, y=Y_S, z=Z_body_i, colorscale=cs_in, opacity=op_in, showscale=False))

                # Cap Start
                r_cap_start = np.linspace(R_inner[0], R_outer[0], 2)
                R_c1, T_c1 = np.meshgrid(r_cap_start, t_surf)
                frame_data.append(go.Surface(
                    x=axis_value + R_c1*np.cos(T_c1),
                    y=np.full_like(R_c1, t_min),
                    z=R_c1*np.sin(T_c1),
                    colorscale=cs_in, opacity=op_in, showscale=False))
                
                # Cap End
                r_cap_end = np.linspace(R_inner[-1], R_outer[-1], 2)
                R_c2, T_c2 = np.meshgrid(r_cap_end, t_surf)
                frame_data.append(go.Surface(
                    x=axis_value + R_c2*np.cos(T_c2),
                    y=np.full_like(R_c2, t_max),
                    z=R_c2*np.sin(T_c2),
                    colorscale=cs_in, opacity=op_in, showscale=False))
            else:
                frame_data.append(go.Surface(x=[], y=[], z=[]))
                frame_data.append(go.Surface(x=[], y=[], z=[]))
                frame_data.append(go.Surface(x=[], y=[], z=[]))
                frame_data.append(go.Surface(x=[], y=[], z=[]))
            
            frames.append(go.Frame(data=frame_data, name=f'rot{k}'))
        
        fig.frames = frames
        
        max_R = max(np.max(np.abs(R_outer)), np.max(np.abs(R_inner))) if len(R_outer) > 0 else 1.0
        x_lo = min(axis_value - max_R*1.3, min(np.min(f_outer), np.min(f_inner)) - 0.2)
        x_hi = max(axis_value + max_R*1.3, max(np.max(f_outer), np.max(f_inner)) + 0.2)
        z_lim = [-max_R*1.3, max_R*1.3]

        fig.update_layout(
            title=f"Washer Method: {func_out_str} vs {func_in_str}, rotated about x = {axis_value}",
            height=600,
            template="plotly_white",
            margin=dict(t=100, b=0, l=0, r=0),
            uirevision='constant',
            updatemenus=[dict(
                type="buttons", direction="left", x=0, y=1.1,
                buttons=[
                    dict(label="Show Washers", method="update", args=[{"visible": vis_washers}]),
                    dict(label="Show Solid", method="update", args=[{"visible": vis_solid}])
                ]
            )],
            scene=dict(
                dragmode="orbit",
                xaxis=dict(title='x', range=[x_lo, x_hi]),
                yaxis=dict(title='y', range=[t_min, t_max]),
                zaxis=dict(title='', range=z_lim, showticklabels=False),
                aspectmode='cube',
                camera=dict(projection=dict(type="orthographic"), eye=dict(x=0.3, y=0.3, z=1.8), up=dict(x=0, y=1, z=0))
            ),
            sliders=[dict(
                steps=[dict(method='animate', args=[[f'rot{k}'], dict(mode='immediate', frame=dict(duration=20, redraw=True))], label='') for k in range(len(steps))],
                currentvalue=dict(visible=False)
            )]
        )

    return fig


# --- MAIN APP LOGIC ---

if method == "Disk Method":
    st.sidebar.subheader("Disk Configuration")
    
    # Axis of Rotation
    axis_type = st.sidebar.selectbox("Axis of Rotation", ["y = c  (horizontal)", "x = c  (vertical)"], key="disk_axis_type")
    axis_value = st.sidebar.number_input("Axis constant c", value=0.0, step=0.5, key="disk_axis_value")
    
    is_horizontal = axis_type.startswith("y")
    
    st.sidebar.markdown("---")
    
    # Function Input — label changes based on axis type
    if is_horizontal:
        func_str = st.sidebar.text_input("Function y =", value="sqrt(x)", key="disk_func")
        c1, c2 = st.sidebar.columns(2)
        t_min = c1.number_input("X Start", value=0.0, key="disk_tmin")
        t_max = c2.number_input("X End", value=4.0, key="disk_tmax")
        st.sidebar.markdown("Add X-values where you want to visualize a disk.")
        points_str = st.sidebar.text_input("X Points (comma separated)", value="1.0, 1.5, 2.0", key="disk_pts")
    else:
        func_str = st.sidebar.text_input("Function x =", value="y^2", key="disk_func")
        c1, c2 = st.sidebar.columns(2)
        t_min = c1.number_input("Y Start", value=0.0, key="disk_tmin")
        t_max = c2.number_input("Y End", value=2.0, key="disk_tmax")
        st.sidebar.markdown("Add Y-values where you want to visualize a disk.")
        points_str = st.sidebar.text_input("Y Points (comma separated)", value="0.5, 1.0, 1.5", key="disk_pts")
    
    try:
        t_points = [float(x.strip()) for x in points_str.split(',') if x.strip()]
    except ValueError:
        st.error("Invalid input. Please enter numbers separated by commas.")
        t_points = [1.0]
    
    axis_t = "y = c" if is_horizontal else "x = c"
    fig = get_disk_method_fig(func_str, t_min, t_max, t_points, axis_t, axis_value)
    st.plotly_chart(fig, use_container_width=True, key="disk_plot")
    
    if is_horizontal:
        st.info("💡 **Instructions:** Define your function y = f(x) (e.g., `sin(x)`, `x^2`, `exp(x)`). Set the axis of rotation y = c. Click 'Play' on the slider to rotate.")
    else:
        st.info("💡 **Instructions:** Define your function x = g(y) (e.g., `y^2`, `sqrt(y)`). Set the axis of rotation x = c. Click 'Play' on the slider to rotate.")

elif method == "Washer Method":
    st.sidebar.subheader("Washer Configuration")
    
    # Axis of Rotation
    axis_type = st.sidebar.selectbox("Axis of Rotation", ["y = c  (horizontal)", "x = c  (vertical)"], key="washer_axis_type")
    axis_value = st.sidebar.number_input("Axis constant c", value=0.0, step=0.5, key="washer_axis_value")
    
    is_horizontal = axis_type.startswith("y")
    
    st.sidebar.markdown("---")
    
    if is_horizontal:
        func_out = st.sidebar.text_input("Outer Function y =", value="sqrt(x)", key="wash_fout")
        func_in = st.sidebar.text_input("Inner Function y =", value="x^2", key="wash_fin")
        c1, c2 = st.sidebar.columns(2)
        t_min = c1.number_input("X Start", value=0.0, key="wash_tmin")
        t_max = c2.number_input("X End", value=1.0, key="wash_tmax")
        st.sidebar.markdown("Add X-values for washer slices.")
        points_str = st.sidebar.text_input("X Points (comma separated)", value="0.2, 0.5, 0.8", key="wash_pts")
    else:
        func_out = st.sidebar.text_input("Outer Function x =", value="sqrt(y)", key="wash_fout")
        func_in = st.sidebar.text_input("Inner Function x =", value="y^2", key="wash_fin")
        c1, c2 = st.sidebar.columns(2)
        t_min = c1.number_input("Y Start", value=0.0, key="wash_tmin")
        t_max = c2.number_input("Y End", value=1.0, key="wash_tmax")
        st.sidebar.markdown("Add Y-values for washer slices.")
        points_str = st.sidebar.text_input("Y Points (comma separated)", value="0.2, 0.5, 0.8", key="wash_pts")
    
    try:
        t_points = [float(x.strip()) for x in points_str.split(',') if x.strip()]
    except ValueError:
        st.error("Invalid input.")
        t_points = [0.5]

    axis_t = "y = c" if is_horizontal else "x = c"
    fig = get_washer_method_fig(func_out, func_in, t_min, t_max, t_points, axis_t, axis_value)
    st.plotly_chart(fig, use_container_width=True, key="washer_plot")
    
    if is_horizontal:
        st.info("💡 **Instructions:** Define Outer/Inner y = f(x). Set the axis of rotation y = c. Click 'Play' to rotate.")
    else:
        st.info("💡 **Instructions:** Define Outer/Inner x = g(y). Set the axis of rotation x = c. Click 'Play' to rotate.")