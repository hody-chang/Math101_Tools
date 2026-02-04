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
def safe_evaluate(func_str, x_arr):
    """
    Evaluates a mathematical string safely using numpy.
    Handles 'x' as the variable and common math functions.
    """
    # Allow students to use ^ for power instead of **
    func_str = func_str.replace("^", "**")
    
    allowed_locals = {
        "x": x_arr,
        "sin": np.sin, "cos": np.cos, "tan": np.tan,
        "sqrt": np.sqrt, "exp": np.exp, "log": np.log, "ln": np.log,
        "pi": np.pi, "e": np.e, "abs": np.abs,
        "power": np.power
    }
    
    try:
        # Evaluate using restricted scope
        return eval(func_str, {"__builtins__": {}}, allowed_locals)
    except Exception as e:
        return None

# --- SIDEBAR CONTROLS ---
st.sidebar.header("Settings")
method = st.sidebar.radio("Choose Method", ["Disk Method", "Washer Method"])

# --- VISUALIZATION FUNCTIONS ---

def get_disk_method_fig(func_str, x_min, x_max, x_points_input):
    # 1. Setup Data
    x_range = np.linspace(x_min, x_max, 100)
    
    # Evaluate User Function
    y_curve = safe_evaluate(func_str, x_range)
    if y_curve is None:
        return go.Figure().add_annotation(text="Invalid Function Syntax", showarrow=False)
        
    z_curve = np.zeros_like(x_range)
    
    # 2. Initialize Figure
    fig = go.Figure()

    # Axis and Curve
    # Adjust axis length based on user input
    padding = (x_max - x_min) * 0.1
    fig.add_trace(go.Scatter3d(x=[x_min-padding, x_max+padding], y=[0, 0], z=[0, 0], mode='lines', line=dict(color='black', width=3), name='Axis'))
    fig.add_trace(go.Scatter3d(x=x_range, y=z_curve, z=y_curve, mode='lines', line=dict(color='blue', width=5), name=f'y={func_str}'))

    # Points
    xp = np.array(x_points_input)
    # Filter points to be within range
    xp = xp[(xp >= x_min) & (xp <= x_max)]
    
    if len(xp) > 0:
        yp = safe_evaluate(func_str, xp)
        zp = np.zeros_like(xp)
        fig.add_trace(go.Scatter3d(x=xp, y=zp, z=yp, mode='markers', marker=dict(size=6, color='red'), name='Points'))

    # Placeholder Traces for Disks (Rim + Face per point)
    for x_val in xp:
        fig.add_trace(go.Scatter3d(x=[], y=[], z=[], mode='lines', line=dict(color='red', width=3), showlegend=False))
        fig.add_trace(go.Mesh3d(x=[], y=[], z=[], color='red', opacity=0.5, showscale=False))

    # Placeholder for Solid Surface
    fig.add_trace(go.Surface(x=[], y=[], z=[], colorscale='Viridis', opacity=0.8, showscale=False, name='Solid', visible=False))
    # Placeholders for Caps (Start and End to make it solid)
    fig.add_trace(go.Surface(x=[], y=[], z=[], colorscale='Viridis', opacity=0.8, showscale=False, name='Cap Start', visible=False))
    fig.add_trace(go.Surface(x=[], y=[], z=[], colorscale='Viridis', opacity=0.8, showscale=False, name='Cap End', visible=False))

    # Calculate Visibility Logic
    num_disk_traces = 2 * len(xp)
    # Total traces: Axis(1) + Curve(1) + Points(1) + Disks(num_disk_traces) + Solid(1) + Caps(2)
    # Note: 3 initial traces (Axis, Curve, Points) + num_disk_traces + 3 solid components
    
    vis_disks = [True] * (3 + num_disk_traces) + [False]*3
    vis_solid = [True, True] + [False]*(1 + num_disk_traces) + [True, True, True]

    # Animation Frames
    frames = []
    steps = np.linspace(0, 2*np.pi, 36)
    
    for k, angle in enumerate(steps):
        frame_data = []
        is_last = (k == len(steps) - 1)
        
        # Static elements
        frame_data.append(go.Scatter3d(x=[x_min-padding, x_max+padding], y=[0, 0], z=[0, 0]))
        frame_data.append(go.Scatter3d(x=x_range, y=z_curve, z=y_curve))
        
        # Rotating Points
        if len(xp) > 0:
            y_rot = yp * np.cos(angle)
            z_rot = yp * np.sin(angle)
            frame_data.append(go.Scatter3d(x=xp, y=z_rot, z=y_rot)) # Swapped
        else:
            frame_data.append(go.Scatter3d(x=[], y=[], z=[]))
        
        # Disks
        for i, x_val in enumerate(xp):
            y_rad = abs(yp[i]) # Radius must be positive
            num_pts = max(2, int(30 * (angle/(2*np.pi))) + 2)
            t_slice = np.linspace(0, angle, num_pts)
            
            y_c = y_rad * np.cos(t_slice)
            z_c = y_rad * np.sin(t_slice)
            x_c = np.full_like(t_slice, x_val)
            
            # Rim (Swapped y/z)
            frame_data.append(go.Scatter3d(x=x_c, y=z_c, z=y_c))
            
            # Face
            x_m = np.concatenate(([x_val], x_c))
            y_m = np.concatenate(([0], z_c))
            z_m = np.concatenate(([0], y_c))
            
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
            X_S, T_S = np.meshgrid(x_range, t_slice)
            R = np.abs(y_curve) 
            
            # Main Body
            frame_data.append(go.Surface(
                x=X_S, y=R*np.sin(T_S), z=R*np.cos(T_S), 
                colorscale='Viridis', opacity=op_solid
            ))

            # Cap Start (x_min)
            r_start = np.linspace(0, np.abs(y_curve[0]), 2)
            R_c1, T_c1 = np.meshgrid(r_start, t_slice)
            frame_data.append(go.Surface(
                x=np.full_like(R_c1, x_min), y=R_c1*np.sin(T_c1), z=R_c1*np.cos(T_c1),
                colorscale='Viridis', opacity=op_solid, showscale=False
            ))

            # Cap End (x_max)
            r_end = np.linspace(0, np.abs(y_curve[-1]), 2)
            R_c2, T_c2 = np.meshgrid(r_end, t_slice)
            frame_data.append(go.Surface(
                x=np.full_like(R_c2, x_max), y=R_c2*np.sin(T_c2), z=R_c2*np.cos(T_c2),
                colorscale='Viridis', opacity=op_solid, showscale=False
            ))

        else:
            frame_data.append(go.Surface(x=[], y=[], z=[]))
            frame_data.append(go.Surface(x=[], y=[], z=[]))
            frame_data.append(go.Surface(x=[], y=[], z=[]))
            
        frames.append(go.Frame(data=frame_data, name=f'rot{k}'))
    
    fig.frames = frames
    
    # Calculate bounds for consistent view
    y_max_val = np.max(np.abs(y_curve)) if len(y_curve) > 0 else 1.0
    y_lim = [-y_max_val*1.2, y_max_val*1.2]

    # Layout
    fig.update_layout(
        title=f"Disk Method: y = {func_str}",
        height=600,
        template="plotly_white", 
        margin=dict(t=100, b=0, l=0, r=0),
        uirevision=True, # Preserve UI state (camera) across updates
        updatemenus=[dict(
            type="buttons", direction="left", x=0, y=1.1,
            buttons=[
                dict(label="Show Disks", method="update", args=[{"visible": vis_disks}]),
                dict(label="Show Solid", method="update", args=[{"visible": vis_solid}])
            ]
        )],
        scene=dict(
            dragmode="orbit", # Allow free rotation
            xaxis=dict(title='X', range=[x_min, x_max]),
            yaxis=dict(title='', range=y_lim, showticklabels=False),
            zaxis=dict(title='', range=y_lim, showticklabels=False),
            aspectmode='cube',
            camera=dict(projection=dict(type="orthographic"), eye=dict(x=1.5, y=1.5, z=1.5), up=dict(x=0, y=0, z=1))
        ),
        sliders=[dict(
            steps=[dict(method='animate', args=[[f'rot{k}'], dict(mode='immediate', frame=dict(duration=20, redraw=True))], label='') for k in range(len(steps))],
            currentvalue=dict(visible=False)
        )]
    )
    return fig

def get_washer_method_fig(func_out_str, func_in_str, x_min, x_max, x_points_input):
    # 1. Setup Data
    x_range = np.linspace(x_min, x_max, 100)
    
    y_outer = safe_evaluate(func_out_str, x_range)
    y_inner = safe_evaluate(func_in_str, x_range)
    
    if y_outer is None or y_inner is None:
        return go.Figure().add_annotation(text="Invalid Function Syntax", showarrow=False)

    zeros = np.zeros_like(x_range)
    
    # 2. Initialize
    fig = go.Figure()

    padding = (x_max - x_min) * 0.1
    # Static Traces
    fig.add_trace(go.Scatter3d(x=[x_min-padding, x_max+padding], y=[0,0], z=[0,0], mode='lines', line=dict(color='black', width=3), name='Axis'))
    fig.add_trace(go.Scatter3d(x=x_range, y=zeros, z=y_outer, mode='lines', line=dict(color='blue', width=4), name=f'Outer: {func_out_str}'))
    fig.add_trace(go.Scatter3d(x=x_range, y=zeros, z=y_inner, mode='lines', line=dict(color='green', width=4), name=f'Inner: {func_in_str}'))

    # Sample Points & Slices
    xp = np.array(x_points_input)
    # Filter points
    xp = xp[(xp >= x_min) & (xp <= x_max)]
    
    if len(xp) > 0:
        yp_out = safe_evaluate(func_out_str, xp)
        yp_in = safe_evaluate(func_in_str, xp)
        
        fig.add_trace(go.Scatter3d(x=np.concatenate([xp,xp]), y=np.zeros(2*len(xp)), z=np.concatenate([yp_out, yp_in]), mode='markers', marker=dict(color='red', size=5), name='Points'))
        
        for i, val in enumerate(xp):
            fig.add_trace(go.Scatter3d(x=[val, val], y=[0, 0], z=[yp_in[i], yp_out[i]], mode='lines', line=dict(color='red', width=4), showlegend=False))
    else:
        yp_out, yp_in = [], []

    # Placeholders for Washers (3 traces per washer: Out, In, Face)
    for _ in xp:
        fig.add_trace(go.Scatter3d(x=[], y=[], z=[], mode='lines', line=dict(color='blue', width=2), showlegend=False))
        fig.add_trace(go.Scatter3d(x=[], y=[], z=[], mode='lines', line=dict(color='green', width=2), showlegend=False))
        fig.add_trace(go.Mesh3d(x=[], y=[], z=[], color='red', opacity=0.5, showscale=False))

    # Placeholders for Surfaces (Outer, Inner)
    fig.add_trace(go.Surface(x=[], y=[], z=[], colorscale='Blues', opacity=0.3, showscale=False, visible=False))
    fig.add_trace(go.Surface(x=[], y=[], z=[], colorscale='Greens', opacity=0.9, showscale=False, visible=False))
    # Placeholders for Caps (Start, End)
    fig.add_trace(go.Surface(x=[], y=[], z=[], colorscale='Greens', opacity=0.9, showscale=False, visible=False))
    fig.add_trace(go.Surface(x=[], y=[], z=[], colorscale='Greens', opacity=0.9, showscale=False, visible=False))

    # Visibility logic
    n_pts = len(xp)
    # Traces: Axis(1) + CurveO(1) + CurveI(1) + Points(1) + Slices(n_pts) + Washers(3*n_pts) + Solids(2) + Caps(2)
    # Total before solids = 4 + n_pts + 3*n_pts
    idx_solids_start = 4 + n_pts + 3*n_pts
    
    # vis_washers: Show up to idx_solids_start, hide last 4
    vis_washers = [True] * idx_solids_start + [False]*4
    
    # vis_solid: Show Axis(1), Curves(2). Hide Points(1), Slices, Washers. Show Solids(2), Caps(2).
    # Hidden items = Points(1) + Slices(n_pts) + Washers(3*n_pts) = 1 + 4*n_pts
    vis_solid = [True, True, True] + [False]*(1 + n_pts + 3*n_pts) + [True, True, True, True]

    # Animation
    frames = []
    steps = np.linspace(0, 2*np.pi, 36)
    
    for k, angle in enumerate(steps):
        frame_data = []
        is_last = (k == len(steps) - 1)
        
        # Static
        frame_data.append(go.Scatter3d(x=[x_min-padding, x_max+padding], y=[0,0], z=[0,0]))
        frame_data.append(go.Scatter3d(x=x_range, y=zeros, z=y_outer))
        frame_data.append(go.Scatter3d(x=x_range, y=zeros, z=y_inner))
        
        # Rotating Markers
        if len(xp) > 0:
            y_mo = yp_out * np.cos(angle)
            z_mo = yp_out * np.sin(angle)
            y_mi = yp_in * np.cos(angle)
            z_mi = yp_in * np.sin(angle)
            
            frame_data.append(go.Scatter3d(x=np.concatenate([xp,xp]), y=np.concatenate([z_mo, z_mi]), z=np.concatenate([y_mo, y_mi])))
            
            for i in range(n_pts):
                frame_data.append(go.Scatter3d(x=[xp[i], xp[i]], y=[z_mi[i], z_mo[i]], z=[y_mi[i], y_mo[i]]))

            for i, val in enumerate(xp):
                num_arc = max(2, int(30 * (angle/(2*np.pi))) + 2)
                t = np.linspace(0, angle, num_arc)
                x_vec = np.full_like(t, val)
                
                yo = yp_out[i] * np.cos(t)
                zo = yp_out[i] * np.sin(t)
                yi = yp_in[i] * np.cos(t)
                zi = yp_in[i] * np.sin(t)
                
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
            frame_data.append(go.Scatter3d(x=[],y=[],z=[])) 
            # Note: The loop for slices/washers won't run if xp is empty

        # Solid + Caps
        if angle > 0:
            t_surf = np.linspace(0, angle, 30)
            X_S, T_S = np.meshgrid(x_range, t_surf)
            
            # Colors
            cs_out = [[0, 'rgb(255, 150, 150)'], [1, 'rgb(255, 150, 150)']] if is_last else 'Blues'
            op_out = 1.0 if is_last else 0.3
            cs_in = [[0, 'rgb(255, 150, 150)'], [1, 'rgb(255, 150, 150)']] if is_last else 'Greens'
            op_in = 1.0 if is_last else 0.9

            # Swapped coords
            R_o = np.tile(np.abs(y_outer), (len(t_surf), 1))
            frame_data.append(go.Surface(x=X_S, y=R_o*np.sin(T_S), z=R_o*np.cos(T_S), colorscale=cs_out, opacity=op_out, showscale=False))
            
            R_i = np.tile(np.abs(y_inner), (len(t_surf), 1))
            frame_data.append(go.Surface(x=X_S, y=R_i*np.sin(T_S), z=R_i*np.cos(T_S), colorscale=cs_in, opacity=op_in, showscale=False))

            # Caps (Start at x_min, End at x_max)
            # Create annulus grid: r goes from y_inner to y_outer
            
            # Cap Start
            r_start = np.linspace(np.abs(y_inner[0]), np.abs(y_outer[0]), 2)
            R_c1, T_c1 = np.meshgrid(r_start, t_surf)
            frame_data.append(go.Surface(
                x=np.full_like(R_c1, x_min), y=R_c1*np.sin(T_c1), z=R_c1*np.cos(T_c1),
                colorscale=cs_in, opacity=op_in, showscale=False
            ))
            
            # Cap End
            r_end = np.linspace(np.abs(y_inner[-1]), np.abs(y_outer[-1]), 2)
            R_c2, T_c2 = np.meshgrid(r_end, t_surf)
            frame_data.append(go.Surface(
                x=np.full_like(R_c2, x_max), y=R_c2*np.sin(T_c2), z=R_c2*np.cos(T_c2),
                colorscale=cs_in, opacity=op_in, showscale=False
            ))

        else:
            frame_data.append(go.Surface(x=[],y=[],z=[]))
            frame_data.append(go.Surface(x=[],y=[],z=[]))
            frame_data.append(go.Surface(x=[],y=[],z=[]))
            frame_data.append(go.Surface(x=[],y=[],z=[]))

        frames.append(go.Frame(data=frame_data, name=f'rot{k}'))
        
    fig.frames = frames
    
    # Calculate bounds
    max_val = max(np.max(np.abs(y_outer)), np.max(np.abs(y_inner))) if len(y_outer) > 0 else 1.0
    y_lim = [-max_val*1.2, max_val*1.2]

    fig.update_layout(
        title=f"Washer Method: {func_out_str} vs {func_in_str}",
        height=600,
        template="plotly_white",
        margin=dict(t=100, b=0, l=0, r=0), 
        uirevision=True, # Preserve UI state (camera) across updates
        updatemenus=[dict(
            type="buttons", direction="left", x=0, y=1.1,
            buttons=[
                dict(label="Show Washers", method="update", args=[{"visible": vis_washers}]),
                dict(label="Show Solid", method="update", args=[{"visible": vis_solid}])
            ]
        )],
        scene=dict(
            dragmode="orbit", # Allow free rotation
            xaxis=dict(title='X', range=[x_min, x_max]),
            yaxis=dict(title='', range=y_lim, showticklabels=False),
            zaxis=dict(title='', range=y_lim, showticklabels=False),
            aspectmode='cube',
            camera=dict(projection=dict(type="orthographic"), eye=dict(x=1.5, y=1.5, z=1.5), up=dict(x=0, y=0, z=1))
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
    
    # Function Input
    func_str = st.sidebar.text_input("Function y =", value="sqrt(x)")
    
    # Range Input
    c1, c2 = st.sidebar.columns(2)
    x_min = c1.number_input("X Start", value=0.0)
    x_max = c2.number_input("X End", value=4.0)
    
    st.sidebar.markdown("Add X-values where you want to visualize a disk.")
    points_str = st.sidebar.text_input("X Points (comma separated)", value="1.0, 1.5, 2.0")
    
    try:
        x_points = [float(x.strip()) for x in points_str.split(',') if x.strip()]
    except ValueError:
        st.error("Invalid input. Please enter numbers separated by commas.")
        x_points = [1.0]
        
    fig = get_disk_method_fig(func_str, x_min, x_max, x_points)
    # Use key to persist component identity (helps with camera state)
    st.plotly_chart(fig, use_container_width=True, key="disk_plot")
    
    st.info("💡 **Instructions:** Define your function (e.g., `sin(x)`, `x^2`, `exp(x)`). Click 'Play' on the slider to rotate the points.")

elif method == "Washer Method":
    st.sidebar.subheader("Washer Configuration")
    
    func_out = st.sidebar.text_input("Outer Radius y =", value="sqrt(x)")
    func_in = st.sidebar.text_input("Inner Radius y =", value="x^2")
    
    c1, c2 = st.sidebar.columns(2)
    x_min = c1.number_input("X Start", value=0.0)
    x_max = c2.number_input("X End", value=1.0)
    
    st.sidebar.markdown("Add X-values for washer slices.")
    points_str = st.sidebar.text_input("X Points (comma separated)", value="0.2, 0.5, 0.8")
    
    try:
        x_points = [float(x.strip()) for x in points_str.split(',') if x.strip()]
    except ValueError:
        st.error("Invalid input.")
        x_points = [0.5]

    fig = get_washer_method_fig(func_out, func_in, x_min, x_max, x_points)
    # Use key to persist component identity
    st.plotly_chart(fig, use_container_width=True, key="washer_plot")
    
    st.info("💡 **Instructions:** Define Outer/Inner radii. Click 'Play' to rotate.")