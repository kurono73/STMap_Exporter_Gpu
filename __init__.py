"""
STMap Exporter

"""


import bpy
import os
import re
import sys
import subprocess
import numpy as np
import bl_operators

# ------------------------------------------------------------------------
# GPU Support (Optional CuPy)
# ------------------------------------------------------------------------

GPU_AVAILABLE = False
try:
    import cupy as cp
    GPU_AVAILABLE = True
    GPU_ERROR = None
except ImportError:
    cp = None
    GPU_ERROR = "CuPy not installed."
except Exception as e:
    cp = None
    GPU_ERROR = f"CuPy error: {str(e)}"

def get_compute_module(use_gpu=False):
    if use_gpu and GPU_AVAILABLE and cp is not None:
        return cp, True
    return np, False

def to_cpu(array):
    if GPU_AVAILABLE and isinstance(array, cp.ndarray):
        return cp.asnumpy(array)
    return array

# ------------------------------------------------------------------------
# Constants
# ------------------------------------------------------------------------

EPSILON_ZERO = 1e-10
EPSILON_DENOM = 1e-12
NEWTON_EPSILON = 1e-5
NEWTON_ITERATIONS = 20
MAX_RESOLUTION = 20000

# ------------------------------------------------------------------------
# Core Math Functions
# ------------------------------------------------------------------------

def get_distortion_params(tracking_cam):
    model = tracking_cam.distortion_model
    params_raw = {'k1': 0.0, 'k2': 0.0, 'k3': 0.0, 'k4': 0.0, 'p1': 0.0, 'p2': 0.0}
    
    if model == 'POLYNOMIAL':
        params_raw['k1'] = float(getattr(tracking_cam, 'k1', 0.0))
        params_raw['k2'] = float(getattr(tracking_cam, 'k2', 0.0))
        params_raw['k3'] = float(getattr(tracking_cam, 'k3', 0.0))
    elif model == 'DIVISION':
        params_raw['k1'] = float(getattr(tracking_cam, 'division_k1', 0.0))
        params_raw['k2'] = float(getattr(tracking_cam, 'division_k2', 0.0))
    elif model == 'NUKE':
        params_raw['k1'] = float(getattr(tracking_cam, 'nuke_k1', 0.0))
        params_raw['k2'] = float(getattr(tracking_cam, 'nuke_k2', 0.0))
        params_raw['p1'] = float(getattr(tracking_cam, 'nuke_p1', 0.0))
        params_raw['p2'] = float(getattr(tracking_cam, 'nuke_p2', 0.0))
    elif model == 'BROWN':
        params_raw['k1'] = float(getattr(tracking_cam, 'brown_k1', 0.0))
        params_raw['k2'] = float(getattr(tracking_cam, 'brown_k2', 0.0))
        params_raw['k3'] = float(getattr(tracking_cam, 'brown_k3', 0.0))
        params_raw['k4'] = float(getattr(tracking_cam, 'brown_k4', 0.0))
        params_raw['p1'] = float(getattr(tracking_cam, 'brown_p1', 0.0))
        params_raw['p2'] = float(getattr(tracking_cam, 'brown_p2', 0.0))
        
    params = {k: np.float64(v) for k, v in params_raw.items()}
    return model, params

def is_distortion_zero(params):
    return all(abs(v) < EPSILON_ZERO for v in params.values())

def explicit_nuke_undistort(xd, yd, model, params, xp):
    rd2 = xd**2 + yd**2
    k1 = params.get('k1', 0.0)
    k2 = params.get('k2', 0.0)
    p1 = params.get('p1', 0.0)
    p2 = params.get('p2', 0.0)
    
    radial = 1.0 + k1 * rd2 + k2 * (rd2**2)
    denom_x = radial + p1 * (yd**2)
    denom_y = radial + p2 * (xd**2)
    
    denom_x = xp.where(xp.abs(denom_x) < EPSILON_DENOM, EPSILON_DENOM, denom_x)
    denom_y = xp.where(xp.abs(denom_y) < EPSILON_DENOM, EPSILON_DENOM, denom_y)
    
    return xd / denom_x, yd / denom_y

def explicit_poly_brown_div_distort(xu, yu, model, params, xp):
    ru2 = xu**2 + yu**2
    
    if model == 'POLYNOMIAL':
        k1 = params.get('k1', 0.0)
        k2 = params.get('k2', 0.0)
        k3 = params.get('k3', 0.0)
        radial = 1.0 + k1 * ru2 + k2 * (ru2**2) + k3 * (ru2**3)
        return xu * radial, yu * radial
        
    elif model == 'DIVISION':
        k1 = params.get('k1', 0.0)
        k2 = params.get('k2', 0.0)
        denom = 1.0 + k1 * ru2 + k2 * (ru2**2)
        denom = xp.where(xp.abs(denom) < EPSILON_DENOM, EPSILON_DENOM, denom)
        return xu / denom, yu / denom
        
    elif model == 'BROWN':
        k1 = params.get('k1', 0.0)
        k2 = params.get('k2', 0.0)
        k3 = params.get('k3', 0.0)
        k4 = params.get('k4', 0.0)
        p1 = params.get('p1', 0.0)
        p2 = params.get('p2', 0.0)
        
        radial = 1.0 + k1 * ru2 + k2 * (ru2**2) + k3 * (ru2**3) + k4 * (ru2**4)
        xd = xu * radial + 2.0 * p1 * xu * yu + p2 * (ru2 + 2.0 * xu**2)
        yd = yu * radial + p1 * (ru2 + 2.0 * yu**2) + 2.0 * p2 * xu * yu
        return xd, yd
    
    return xu, yu

def newton_inverse(target_x, target_y, func, model, params, xp, iters=NEWTON_ITERATIONS):
    x_est = xp.copy(target_x)
    y_est = xp.copy(target_y)
    eps = NEWTON_EPSILON
    
    for _ in range(iters):
        fx, fy = func(x_est, y_est, model, params, xp)
        ex = fx - target_x
        ey = fy - target_y
        
        fx_x, fy_x = func(x_est + eps, y_est, model, params, xp)
        Jxx = (fx_x - fx) / eps
        Jyx = (fy_x - fy) / eps
        
        fx_y, fy_y = func(x_est, y_est + eps, model, params, xp)
        Jxy = (fx_y - fx) / eps
        Jyy = (fy_y - fy) / eps
        
        det = Jxx * Jyy - Jxy * Jyx
        det = xp.where(xp.abs(det) < EPSILON_DENOM, xp.copysign(EPSILON_DENOM, det), det)
        
        x_est -= (Jyy * ex - Jxy * ey) / det
        y_est -= (-Jyx * ex + Jxx * ey) / det
    
    return x_est, y_est

def calc_distortion(xu, yu, model, params, use_gpu=False):
    xp, is_gpu = get_compute_module(use_gpu)
    if is_gpu:
        xu, yu = cp.asarray(xu), cp.asarray(yu)
    
    if model == 'NUKE':
        res = newton_inverse(xu, yu, explicit_nuke_undistort, model, params, xp)
    else:
        res = explicit_poly_brown_div_distort(xu, yu, model, params, xp)
        
    return (to_cpu(res[0]), to_cpu(res[1])) if is_gpu else res

def calc_undistortion(xd, yd, model, params, use_gpu=False):
    xp, is_gpu = get_compute_module(use_gpu)
    if is_gpu:
        xd, yd = cp.asarray(xd), cp.asarray(yd)
    
    if model == 'NUKE':
        res = explicit_nuke_undistort(xd, yd, model, params, xp)
    else:
        res = newton_inverse(xd, yd, explicit_poly_brown_div_distort, model, params, xp)
        
    return (to_cpu(res[0]), to_cpu(res[1])) if is_gpu else res

# ------------------------------------------------------------------------
# Utilities
# ------------------------------------------------------------------------

def estimate_auto_bbox(base_w, base_h, cx, cy, fx_math, fy_math, model, params, use_gpu=False):
    samples = 100
    xs_edge = np.concatenate([
        np.linspace(0, base_w, samples), np.linspace(0, base_w, samples),
        np.zeros(samples), np.full(samples, base_w)
    ])
    ys_edge = np.concatenate([
        np.zeros(samples), np.full(samples, base_h),
        np.linspace(0, base_h, samples), np.linspace(0, base_h, samples)
    ])
    
    xd_n = (xs_edge - cx) / fx_math
    yd_n = (ys_edge - cy) / fy_math
    xu_n, yu_n = calc_undistortion(xd_n, yd_n, model, params, use_gpu)
    
    xu_px = xu_n * fx_math + cx
    yu_px = yu_n * fy_math + cy
    
    min_x = int(np.floor(np.min(xu_px))) - 2
    max_x = int(np.ceil(np.max(xu_px))) + 2
    min_y = int(np.floor(np.min(yu_px))) - 2
    max_y = int(np.ceil(np.max(yu_px))) + 2
    
    min_x = min(min_x, 0)
    max_x = max(max_x, int(base_w))
    min_y = min(min_y, 0)
    max_y = max(max_y, int(base_h))
    
    return min_x, max_x, min_y, max_y

def calculate_overscan_dimensions(context, props, clip):
    tracking_cam = clip.tracking.camera
    model, params = get_distortion_params(tracking_cam)
    distortion_is_zero = is_distortion_zero(params)
    
    orig_w = float(clip.size[0])
    orig_h = float(clip.size[1])
    
    base_w = float(props.custom_res_x) if props.use_custom_resolution else orig_w
    base_h = float(props.custom_res_y) if props.use_custom_resolution else orig_h
    
    scale_x = base_w / orig_w
    scale_y = base_h / orig_h
    
    pr_x = float(getattr(tracking_cam, 'principal', [0.0, 0.0])[0]) * scale_x
    pr_y = float(getattr(tracking_cam, 'principal', [0.0, 0.0])[1]) * scale_y
    cx = base_w / 2.0 + pr_x
    cy = base_h / 2.0 + pr_y
    
    cam_sensor_base = float(tracking_cam.sensor_width)
    focal_px = float(tracking_cam.focal_length) * (orig_w / cam_sensor_base)
    pixel_aspect = float(getattr(tracking_cam, 'pixel_aspect', 1.0))
    
    if model == 'NUKE':
        fx_math = base_w / 2.0
        fy_math = (base_w / 2.0) / pixel_aspect
    else:
        fx_math = focal_px * scale_x
        fy_math = (focal_px / pixel_aspect) * scale_y
        
    use_gpu = props.use_gpu and GPU_AVAILABLE
    
    min_x_auto, max_x_auto, min_y_auto, max_y_auto = 0, int(base_w), 0, int(base_h)
    if not distortion_is_zero:
        min_x_auto, max_x_auto, min_y_auto, max_y_auto = estimate_auto_bbox(
            base_w, base_h, cx, cy, fx_math, fy_math, model, params, use_gpu
        )
        lim_x = int(base_w * props.max_overscan_percent / 200.0)
        lim_y = int(base_h * props.max_overscan_percent / 200.0)
        min_x_auto = max(min_x_auto, -lim_x)
        max_x_auto = min(max_x_auto, int(base_w) + lim_x)
        min_y_auto = max(min_y_auto, -lim_y)
        max_y_auto = min(max_y_auto, int(base_h) + lim_y)
        
    pad = props.extra_padding
    min_x_auto -= pad
    max_x_auto += pad
    min_y_auto -= pad
    max_y_auto += pad
    
    auto_w = max_x_auto - min_x_auto
    auto_h = max_y_auto - min_y_auto
    
    if props.overscan_mode == 'CUSTOM':
        out_w = max(props.custom_overscan_x, int(base_w))
        out_h = max(props.custom_overscan_y, int(base_h))
        
        pad_x_total = out_w - int(base_w)
        pad_y_total = out_h - int(base_h)
        min_x = int(-np.floor(pad_x_total / 2.0))
        max_x = min_x + out_w
        min_y = int(-np.floor(pad_y_total / 2.0))
        max_y = min_y + out_h
    else:
        min_x, max_x, min_y, max_y = min_x_auto, max_x_auto, min_y_auto, max_y_auto
        out_w, out_h = auto_w, auto_h
        
    os_sensor_width = cam_sensor_base * (out_w / base_w) if base_w > 0 else cam_sensor_base
        
    return {
        'base_w': base_w, 'base_h': base_h,
        'out_w': out_w, 'out_h': out_h,
        'min_x': min_x, 'max_x': max_x,
        'min_y': min_y, 'max_y': max_y,
        'auto_w': auto_w, 'auto_h': auto_h,
        'cx': cx, 'cy': cy,
        'fx_math': fx_math, 'fy_math': fy_math,
        'model': model, 'params': params,
        'distortion_is_zero': distortion_is_zero,
        'sensor_width': cam_sensor_base,
        'os_sensor_width': os_sensor_width,
        'use_gpu': use_gpu
    }

def get_clean_clip_name(clip_name):
    base = os.path.splitext(clip_name)[0]
    base_clean = re.sub(r'[\._-]?\d+$', '', base)
    return base_clean if base_clean else "STMap"

def generate_checkerboard(width, height, min_x, min_y, base_w, cx, cy, grid_count_x=25):
    grid_size_px = base_w / float(grid_count_x)
    ys, xs = np.mgrid[0:height, 0:width]
    
    xs_global = xs + min_x + 0.5
    ys_global = ys + min_y + 0.5
    xs_centered = xs_global - cx
    ys_centered = ys_global - cy
    
    checker = ((xs_centered // grid_size_px) % 2) != ((ys_centered // grid_size_px) % 2)
    return np.where(checker, 0.5, 0.2).astype(np.float32)

def get_extension_from_format(format_enum):
    return {'OPEN_EXR': 'exr', 'TIFF': 'tif', 'PNG': 'png'}.get(format_enum, 'exr')


# ------------------------------------------------------------------------
# UI Callbacks & Property Updates
# ------------------------------------------------------------------------

def get_clip_from_context(context):
    space = getattr(context, "space_data", None)
    return getattr(space, "clip", None)

def get_base_res(context, self):
    clip = get_clip_from_context(context)
    if not clip: return 1920, 1080
    if self.use_custom_resolution:
        return self.custom_res_x, self.custom_res_y
    return clip.size[0], clip.size[1]

def update_use_custom_res(self, context):
    if self.use_custom_resolution and not self.is_updating:
        clip = get_clip_from_context(context)
        if clip:
            self.is_updating = True
            self.custom_res_x = int(clip.size[0])
            self.custom_res_y = int(clip.size[1])
            self.is_updating = False

def update_custom_res_x(self, context):
    if self.lock_aspect and not self.is_updating:
        clip = get_clip_from_context(context)
        if clip and clip.size[0] > 0:
            self.is_updating = True
            self.custom_res_y = max(1, int(self.custom_res_x * (clip.size[1] / clip.size[0])))
            self.is_updating = False

def update_custom_res_y(self, context):
    if self.lock_aspect and not self.is_updating:
        clip = get_clip_from_context(context)
        if clip and clip.size[1] > 0:
            self.is_updating = True
            self.custom_res_x = max(1, int(self.custom_res_y * (clip.size[0] / clip.size[1])))
            self.is_updating = False

def update_overscan_mode(self, context):
    if self.overscan_mode == 'CUSTOM' and not self.is_updating:
        self.is_updating = True
        base_w, base_h = get_base_res(context, self)
        pct = max(100.0, self.custom_overscan_percent)
        self.custom_overscan_x = max(int(base_w), int(base_w * (pct / 100.0)))
        self.custom_overscan_y = max(int(base_h), int(base_h * (pct / 100.0)))
        self.is_updating = False

def update_overscan_percent(self, context):
    if not self.is_updating:
        self.is_updating = True
        base_w, base_h = get_base_res(context, self)
        pct = max(100.0, self.custom_overscan_percent)
        self.custom_overscan_percent = pct
        self.custom_overscan_x = max(int(base_w), int(base_w * (pct / 100.0)))
        self.custom_overscan_y = max(int(base_h), int(base_h * (pct / 100.0)))
        self.is_updating = False

def update_overscan_x(self, context):
    if not self.is_updating:
        self.is_updating = True
        base_w, base_h = get_base_res(context, self)
        
        if self.custom_overscan_x < base_w:
            self.custom_overscan_x = int(base_w)
            
        if self.lock_overscan_aspect and base_w > 0:
            self.custom_overscan_y = max(int(base_h), int(self.custom_overscan_x * (base_h / base_w)))
            
        self.custom_overscan_percent = (self.custom_overscan_x / base_w) * 100.0 if base_w > 0 else 100.0
        self.is_updating = False

def update_overscan_y(self, context):
    if not self.is_updating:
        self.is_updating = True
        base_w, base_h = get_base_res(context, self)
        
        if self.custom_overscan_y < base_h:
            self.custom_overscan_y = int(base_h)
            
        if self.lock_overscan_aspect and base_h > 0:
            self.custom_overscan_x = max(int(base_w), int(self.custom_overscan_y * (base_w / base_h)))
            self.custom_overscan_percent = (self.custom_overscan_x / base_w) * 100.0 if base_w > 0 else 100.0
        else:
            self.custom_overscan_percent = (self.custom_overscan_y / base_h) * 100.0 if base_h > 0 else 100.0
            
        self.is_updating = False


# ------------------------------------------------------------------------
# Properties
# ------------------------------------------------------------------------

class STMapExporterProperties(bpy.types.PropertyGroup):
    output_folder: bpy.props.StringProperty(
        name="Export Path", description="Directory to export maps. Leave empty to save next to the .blend file",
        subtype='DIR_PATH', default=""
    )
    use_custom_name: bpy.props.BoolProperty(
        name="Use Custom Name", description="Use a custom prefix for the exported files instead of the clip name", default=False
    )
    custom_name: bpy.props.StringProperty(
        name="", description="Custom prefix string for exported files", default=""
    )
    
    export_undistort: bpy.props.BoolProperty(
        name="Undistort Map", description="Export the Undistort STMap (Forward mapping from Distorted to Undistorted)", default=True
    )
    export_redistort: bpy.props.BoolProperty(
        name="Redistort Map", description="Export the Redistort STMap (Inverse mapping from Undistorted to Distorted)", default=True
    )
    export_grids: bpy.props.BoolProperty(
        name="Checkerboard Guides", description="Export grayscale checkerboard grids to visualize the distortion effect", default=False
    )
    grid_count: bpy.props.IntProperty(
        name="Grid Count (X)", description="Number of checker squares along the image width", default=25, min=2, max=128
    )
    
    use_custom_resolution: bpy.props.BoolProperty(
        name="Override Base Res", description="Override the base resolution instead of using the clip's original resolution", default=False, update=update_use_custom_res
    )
    custom_res_x: bpy.props.IntProperty(
        name="Width", description="Custom base output width in pixels", min=1, default=1920, update=update_custom_res_x
    )
    custom_res_y: bpy.props.IntProperty(
        name="Height", description="Custom base output height in pixels", min=1, default=1080, update=update_custom_res_y
    )
    lock_aspect: bpy.props.BoolProperty(
        name="Lock", description="Lock aspect ratio to the original clip's dimensions", default=True
    )
    is_updating: bpy.props.BoolProperty(default=False)
    
    show_overscan: bpy.props.BoolProperty(
        name="Overscan", description="Show Overscan / BBox settings", default=False
    )
    overscan_mode: bpy.props.EnumProperty(
        name="Mode",
        description="Select how to determine the overscan resolution",
        items=[('AUTO', "Auto Expansion", "Automatically calculate bounding box based on distortion"), 
               ('CUSTOM', "Custom Resolution", "Set a specific overscan resolution")],
        default='AUTO',
        update=update_overscan_mode
    )
    custom_overscan_percent: bpy.props.FloatProperty(
        name="Scale (%)", description="Overscan percentage relative to base resolution", min=100.0, default=105.0, update=update_overscan_percent
    )
    custom_overscan_x: bpy.props.IntProperty(
        name="Width", description="Custom overscan width in pixels", min=1, default=2016, update=update_overscan_x
    )
    custom_overscan_y: bpy.props.IntProperty(
        name="Height", description="Custom overscan height in pixels", min=1, default=1134, update=update_overscan_y
    )
    lock_overscan_aspect: bpy.props.BoolProperty(
        name="Lock Aspect", description="Lock aspect ratio when scaling overscan resolution", default=True
    )
    
    max_overscan_percent: bpy.props.IntProperty(
        name="Overscan Limit (%)", description="Maximum allowed auto-expansion relative to the base resolution to prevent OOM errors", default=50, min=0, max=500
    )
    extra_padding: bpy.props.IntProperty(
        name="Extra Padding (px)", description="Additional uniform pixel padding added to the Auto bounding box", default=0, min=0, max=2000
    )

    file_format: bpy.props.EnumProperty(
        name="Format", description="Image format for the exported STMaps (OpenEXR highly recommended)",
        items=[('OPEN_EXR', "OpenEXR", ""), ('TIFF', "TIFF", ""), ('PNG', "PNG", "")], default='OPEN_EXR'
    )
    exr_depth: bpy.props.EnumProperty(
        name="Color Depth", description="Color depth for OpenEXR files (32-bit Float recommended for accurate STMaps)",
        items=[('16', "Float (Half)", ""), ('32', "Float (Full)", "")], default='32'
    )
    exr_codec: bpy.props.EnumProperty(
        name="Codec", description="Compression method for OpenEXR files",
        items=[('NONE', "None", ""), ('PXR24', "Pxr24", ""), ('ZIP', "ZIP", ""), 
               ('PIZ', "PIZ", ""), ('RLE', "RLE", ""), ('ZIPS', "ZIPS", ""), 
               ('DWAA', "DWAA", ""), ('DWAB', "DWAB", "")], default='ZIP'
    )
    std_depth: bpy.props.EnumProperty(
        name="Color Depth", description="Color depth for standard image formats",
        items=[('8', "8-bit", ""), ('16', "16-bit", "")], default='16'
    )
    remap_bbox: bpy.props.BoolProperty(
        name="Remap to [0-1] (BBox)", 
        description="Remap Redistort UVs to a strict 0.0-1.0 range based on the expanded bounding box. Useful for non-floating point formats (PNG/TIFF) to prevent clipping negative values", 
        default=False
    )
    
    show_options: bpy.props.BoolProperty(name="Options", description="Show advanced format and performance settings", default=False)
    use_gpu: bpy.props.BoolProperty(
        name="Use GPU Acceleration", description="Use CUDA GPU for much faster calculations. Falls back to CPU if unavailable", default=GPU_AVAILABLE
    )
    cuda_version: bpy.props.EnumProperty(
        name="CUDA Version", description="Select your NVIDIA driver's CUDA version for CuPy installation",
        items=[('AUTO', "Auto Detect", "Automatically detect CUDA version using nvidia-smi"),
               ('11x', "CUDA 11.x", "For older NVIDIA drivers"),
               ('12x', "CUDA 12.x", "For current NVIDIA drivers"),
               ('13x', "CUDA 13.x", "For newer NVIDIA drivers")],
        default='AUTO'
    )

# ------------------------------------------------------------------------
# Operators
# ------------------------------------------------------------------------

class STMAP_OT_install_cupy(bpy.types.Operator):
    bl_idname = "stmap.install_cupy"
    bl_label = "Install CuPy"
    
    def detect_cuda_version(self):
        try:
            res = subprocess.run(['nvidia-smi'], capture_output=True, text=True, timeout=5)
            if res.returncode == 0:
                match = re.search(r"CUDA Version:\s*(\d+)", res.stdout)
                if match: return f"{match.group(1)}x"
        except (FileNotFoundError, subprocess.TimeoutExpired):
            pass
        return None

    def execute(self, context):
        props = context.scene.stmap_exporter_props
        target_version = props.cuda_version
        
        if target_version == 'AUTO':
            self.report({'INFO'}, "Detecting NVIDIA GPU and CUDA version...")
            detected = self.detect_cuda_version()
            if not detected:
                self.report({'ERROR'}, "Failed to detect GPU. Please select CUDA version manually.")
                return {'CANCELLED'}
            target_version = detected
            self.report({'INFO'}, f"Detected CUDA {target_version.replace('x', '.x')}")

        pkg_name = f"cupy-cuda{target_version}"
        self.report({'INFO'}, f"Installing {pkg_name}... Blender will freeze. Please wait.")
        
        try:
            subprocess.run([sys.executable, "-m", "pip", "install", pkg_name], check=True, capture_output=True, text=True)
            self.report({'INFO'}, f"{pkg_name} installed successfully! Please restart Blender.")
            return {'FINISHED'}
        except subprocess.CalledProcessError as e:
            err = e.stderr[-200:] if e.stderr else "Unknown pip error"
            self.report({'ERROR'}, f"Failed to install {pkg_name}: {err}")
            return {'CANCELLED'}
        except Exception as e:
            self.report({'ERROR'}, f"Error: {str(e)}")
            return {'CANCELLED'}


class STMAP_OT_reset_custom_res(bpy.types.Operator):
    bl_idname = "stmap.reset_custom_res"
    bl_label = "Reset to Original Resolution"
    
    @classmethod
    def poll(cls, context):
        return get_clip_from_context(context) is not None
        
    def execute(self, context):
        clip = get_clip_from_context(context)
        props = context.scene.stmap_exporter_props
        props.is_updating = True
        props.custom_res_x = int(clip.size[0])
        props.custom_res_y = int(clip.size[1])
        props.is_updating = False
        return {'FINISHED'}


class STMAP_OT_copy_clipboard(bpy.types.Operator):
    """Copy value to clipboard"""
    bl_idname = "stmap.copy_clipboard"
    bl_label = "Copy"
    
    text_to_copy: bpy.props.StringProperty()
    message: bpy.props.StringProperty()
    
    def execute(self, context):
        context.window_manager.clipboard = self.text_to_copy
        self.report({'INFO'}, f"Copied to clipboard: {self.message}")
        return {'FINISHED'}


class STMAP_OT_apply_overscan(bpy.types.Operator):
    bl_idname = "stmap.apply_overscan"
    bl_label = "Apply Overscan to Scene"
    bl_description = "Apply overscan resolution to Scene render settings and scale active Camera sensor"
    
    @classmethod
    def poll(cls, context):
        return get_clip_from_context(context) is not None
        
    def execute(self, context):
        clip = get_clip_from_context(context)
        props = context.scene.stmap_exporter_props
        
        try:
            os_data = calculate_overscan_dimensions(context, props, clip)
        except Exception as e:
            self.report({'ERROR'}, f"Calculation failed: {str(e)}")
            return {'CANCELLED'}
            
        scene = context.scene
        scene.render.resolution_x = int(os_data['out_w'])
        scene.render.resolution_y = int(os_data['out_h'])
        
        camera_obj = scene.camera
        if camera_obj and camera_obj.type == 'CAMERA':
            cam_data = camera_obj.data
            cam_data.sensor_width = os_data['os_sensor_width']
            self.report({'INFO'}, f"Applied: Res {os_data['out_w']}x{os_data['out_h']}, Sensor {cam_data.sensor_width:.2f}mm to '{camera_obj.name}'")
        else:
            self.report({'WARNING'}, f"Applied Res {os_data['out_w']}x{os_data['out_h']}. No active scene camera found.")
            
        return {'FINISHED'}


class STMAP_OT_restore_overscan(bpy.types.Operator):
    bl_idname = "stmap.restore_overscan"
    bl_label = "Restore Scene Res"
    bl_description = "Restore Scene render resolution and camera sensor to base footage values"
    
    @classmethod
    def poll(cls, context):
        return get_clip_from_context(context) is not None
        
    def execute(self, context):
        clip = get_clip_from_context(context)
        props = context.scene.stmap_exporter_props
        
        try:
            os_data = calculate_overscan_dimensions(context, props, clip)
        except Exception as e:
            self.report({'ERROR'}, f"Calculation failed: {str(e)}")
            return {'CANCELLED'}
            
        scene = context.scene
        scene.render.resolution_x = int(os_data['base_w'])
        scene.render.resolution_y = int(os_data['base_h'])
        
        camera_obj = scene.camera
        if camera_obj and camera_obj.type == 'CAMERA':
            cam_data = camera_obj.data
            cam_data.sensor_width = os_data['sensor_width']
            self.report({'INFO'}, f"Restored: Res {int(os_data['base_w'])}x{int(os_data['base_h'])}, Sensor {cam_data.sensor_width:.2f}mm on '{camera_obj.name}'")
        else:
            self.report({'WARNING'}, f"Restored Res {int(os_data['base_w'])}x{int(os_data['base_h'])}.")
            
        return {'FINISHED'}


class STMAP_OT_export(bpy.types.Operator):
    bl_idname = "stmap.export"
    bl_label = "Export Maps"
    bl_description = "Calculate and export selected STMaps and Checker Grids"
    
    @classmethod
    def poll(cls, context):
        return get_clip_from_context(context) is not None
    
    def execute(self, context):
        import time
        start_time = time.time()
        
        props = context.scene.stmap_exporter_props
        if props.use_gpu and not GPU_AVAILABLE:
            self.report({'WARNING'}, f"GPU unavailable: {GPU_ERROR}. Using CPU.")
            props.use_gpu = False
            
        ep = self.prepare_export_parameters(context)
        if not ep: return {'CANCELLED'}
        
        results = self.execute_export(ep, context)
        
        elapsed = time.time() - start_time
        self.report_results(results, ep['distortion_is_zero'], elapsed, ep['use_gpu'])
        return {'FINISHED'}
    
    def prepare_export_parameters(self, context):
        clip = get_clip_from_context(context)
        props = context.scene.stmap_exporter_props
        
        try:
            ep = calculate_overscan_dimensions(context, props, clip)
        except Exception as e:
            self.report({'ERROR'}, f"Failed to calculate dimensions: {str(e)}")
            return None
            
        if ep['out_w'] <= 0 or ep['out_h'] <= 0 or ep['out_w'] > MAX_RESOLUTION or ep['out_h'] > MAX_RESOLUTION:
            self.report({'ERROR'}, "Invalid BBox or resolution too high.")
            return None
            
        raw_path = props.output_folder.strip()
        if not raw_path:
            raw_path = "//"
            
        if raw_path.startswith("//") and not bpy.data.filepath:
            self.report({'ERROR'}, "Please save your .blend file first, or specify an absolute export path.")
            return None
            
        out_dir = bpy.path.abspath(raw_path)
        try:
            os.makedirs(out_dir, exist_ok=True)
        except Exception as e:
            self.report({'ERROR'}, f"Failed to create directory: {str(e)}")
            return None
            
        prefix = props.custom_name.strip() if props.use_custom_name and props.custom_name.strip() else get_clean_clip_name(clip.name)
        
        ep['output_dir'] = out_dir
        ep['prefix'] = prefix
        ep['props'] = props
        ep['ext'] = get_extension_from_format(props.file_format)
        
        return ep
    
    def execute_export(self, ep, context):
        msgs = []
        if ep['props'].export_undistort:
            self.export_map(ep, context, is_undistort=True)
            msgs.append("Undistort Map")
        if ep['props'].export_redistort:
            self.export_map(ep, context, is_undistort=False)
            msgs.append("Redistort Map")
        if ep['props'].export_grids:
            self.export_grids(ep)
            msgs.append("Checker Grids")
        return msgs

    def export_map(self, ep, context, is_undistort):
        w, h = (ep['out_w'], ep['out_h']) if is_undistort else (int(ep['base_w']), int(ep['base_h']))
        min_x, min_y = (ep['min_x'], ep['min_y']) if is_undistort else (0, 0)
        
        ys, xs = np.mgrid[0:h, 0:w].astype(np.float64)
        xn = (xs + min_x + 0.5 - ep['cx']) / ep['fx_math']
        yn = (ys + min_y + 0.5 - ep['cy']) / ep['fy_math']
        
        if is_undistort:
            x_res, y_res = calc_distortion(xn, yn, ep['model'], ep['params'], ep['use_gpu'])
        else:
            x_res, y_res = calc_undistortion(xn, yn, ep['model'], ep['params'], ep['use_gpu'])
            
        px = x_res * ep['fx_math'] + ep['cx']
        py = y_res * ep['fy_math'] + ep['cy']
        
        if is_undistort:
            map_x = (px / ep['base_w']).astype(np.float32)
            map_y = (py / ep['base_h']).astype(np.float32)
        else:
            if ep['props'].remap_bbox:
                map_x = ((px - ep['min_x']) / ep['out_w']).astype(np.float32)
                map_y = ((py - ep['min_y']) / ep['out_h']).astype(np.float32)
            else:
                map_x = (px / ep['base_w']).astype(np.float32)
                map_y = (py / ep['base_h']).astype(np.float32)
        
        name = "Undistort" if is_undistort else "Redistort"
        filepath = os.path.join(ep['output_dir'], f"{ep['prefix']}_{name}_STMap.{ep['ext']}")
        self.save_image(filepath, map_x, map_y, w, h, ep['props'], context.scene)

    def export_grids(self, ep):
        chk = generate_checkerboard(ep['out_w'], ep['out_h'], ep['min_x'], ep['min_y'], ep['base_w'], ep['cx'], ep['cy'], ep['props'].grid_count)
        base_w_int, base_h_int = int(ep['base_w']), int(ep['base_h'])
        
        grid_st = np.zeros((base_h_int, base_w_int), dtype=np.float32)
        c_x1, c_y1 = max(0, -ep['min_x']), max(0, -ep['min_y'])
        c_x2, c_y2 = min(ep['out_w'], c_x1 + base_w_int), min(ep['out_h'], c_y1 + base_h_int)
        d_x1, d_y1 = max(0, ep['min_x']), max(0, ep['min_y'])
        
        if c_x2 > c_x1 and c_y2 > c_y1:
            grid_st[d_y1:d_y1+(c_y2-c_y1), d_x1:d_x1+(c_x2-c_x1)] = chk[c_y1:c_y2, c_x1:c_x2]
            
        self.save_png(os.path.join(ep['output_dir'], f"{ep['prefix']}_Grid_Undistorted.png"), grid_st, base_w_int, base_h_int)
        
        ys, xs = np.mgrid[0:base_h_int, 0:base_w_int].astype(np.float64)
        xn = (xs + 0.5 - ep['cx']) / ep['fx_math']
        yn = (ys + 0.5 - ep['cy']) / ep['fy_math']
        
        xu, yu = calc_undistortion(xn, yn, ep['model'], ep['params'], ep['use_gpu'])
        px, py = xu * ep['fx_math'] + ep['cx'], yu * ep['fy_math'] + ep['cy']
        
        map_x = np.clip(np.floor(px - ep['min_x']).astype(int), 0, ep['out_w'] - 1)
        map_y = np.clip(np.floor(py - ep['min_y']).astype(int), 0, ep['out_h'] - 1)
        
        self.save_png(os.path.join(ep['output_dir'], f"{ep['prefix']}_Grid_Distorted.png"), chk[map_y, map_x], base_w_int, base_h_int)

    def report_results(self, msgs, is_zero, elapsed, use_gpu):
        if msgs:
            msg = f"Exported: {', '.join(msgs)} [{'GPU' if use_gpu else 'CPU'}, {elapsed:.2f}s]"
            if is_zero: msg += " (Zero distortion)"
            self.report({'WARNING' if is_zero else 'INFO'}, msg)
        else:
            self.report({'WARNING'}, "No export options selected.")

    def save_image(self, fp, u, v, w, h, props, scene):
        is_float = props.file_format == 'OPEN_EXR'
        img = bpy.data.images.new("TMP", w, h, alpha=False, float_buffer=is_float)
        img.colorspace_settings.name = 'Non-Color'
        pixels = np.zeros((h, w, 4), dtype=np.float32)
        pixels[..., 0], pixels[..., 1], pixels[..., 3] = u, v, 1.0
        img.pixels.foreach_set(pixels.ravel())
        
        rnd = scene.render.image_settings
        orig = (rnd.file_format, rnd.color_depth, rnd.exr_codec, scene.view_settings.view_transform)
        try:
            rnd.file_format = props.file_format
            if is_float: rnd.color_depth, rnd.exr_codec = props.exr_depth, props.exr_codec
            else: rnd.color_depth = props.std_depth
            scene.view_settings.view_transform = 'Raw'
            img.save_render(fp, scene=scene)
        finally:
            rnd.file_format, rnd.color_depth, rnd.exr_codec, scene.view_settings.view_transform = orig
            bpy.data.images.remove(img)
            
    def save_png(self, fp, gray, w, h):
        img = bpy.data.images.new("TMP", w, h, alpha=False, float_buffer=False)
        pixels = np.zeros((h, w, 4), dtype=np.float32)
        pixels[..., 0], pixels[..., 1], pixels[..., 2], pixels[..., 3] = gray, gray, gray, 1.0
        img.pixels.foreach_set(pixels.ravel())
        img.file_format, img.filepath_raw = 'PNG', fp
        img.save()
        bpy.data.images.remove(img)

# ------------------------------------------------------------------------
# Presets
# ------------------------------------------------------------------------

class STMAP_MT_presets(bpy.types.Menu):
    bl_label = "STMap Presets"
    bl_idname = "STMAP_MT_presets"
    preset_subdir = "stmap_exporter"
    preset_operator = "script.execute_preset"
    draw = bpy.types.Menu.draw_preset


class STMAP_OT_add_preset(bl_operators.presets.AddPresetBase, bpy.types.Operator):
    bl_idname = "stmap.preset_add"
    bl_label = "Add STMap Preset"
    preset_menu = "STMAP_MT_presets"
    
    preset_defines = [
        "props = bpy.context.scene.stmap_exporter_props"
    ]
    
    preset_values = [
        "props.export_undistort",
        "props.export_redistort",
        "props.export_grids",
        "props.grid_count",
        "props.use_custom_resolution",
        "props.custom_res_x",
        "props.custom_res_y",
        "props.lock_aspect",
        "props.overscan_mode",
        "props.custom_overscan_percent",
        "props.custom_overscan_x",
        "props.custom_overscan_y",
        "props.lock_overscan_aspect",
        "props.max_overscan_percent",
        "props.extra_padding",
        "props.file_format",
        "props.exr_depth",
        "props.exr_codec",
        "props.std_depth",
        "props.remap_bbox",
        "props.use_gpu"
    ]
    preset_subdir = "stmap_exporter"


class STMAP_PT_presets(bpy.types.Panel):
    bl_label = "STMap Presets"
    bl_idname = "STMAP_PT_presets"
    bl_space_type = 'CLIP_EDITOR'
    bl_region_type = 'HEADER'
    
    preset_subdir = "stmap_exporter"
    preset_operator = "script.execute_preset"
    preset_add_operator = "stmap.preset_add"
    path_menu = bpy.types.Menu.path_menu
    
    def draw(self, context):
        layout = self.layout
        layout.emboss = 'PULLDOWN_MENU'
        layout.operator_context = 'EXEC_DEFAULT'
        bpy.types.Menu.draw_preset(self, context)


# ------------------------------------------------------------------------
# UI
# ------------------------------------------------------------------------

class STMAP_PT_main_panel(bpy.types.Panel):
    bl_label = "STMap Exporter"
    bl_idname = "STMAP_PT_main_panel"
    bl_space_type = 'CLIP_EDITOR'
    bl_region_type = 'UI'
    bl_category = 'Track'
    
    @classmethod
    def poll(cls, context):
        return get_clip_from_context(context) is not None

    def draw_header_preset(self, context):
        layout = self.layout
        layout.emboss = 'NONE'
        layout.popover(
            panel="STMAP_PT_presets",
            icon='PRESET',
            text=""
        )
    
    def draw(self, context):
        layout = self.layout
        clip = get_clip_from_context(context)
        props = context.scene.stmap_exporter_props
        
        _, params = get_distortion_params(clip.tracking.camera)
        
        row = layout.row()
        row.scale_y = 1.5
        row.operator("stmap.export", icon='IMAGE_DATA')
        layout.separator()
        
        if is_distortion_zero(params):
            b = layout.box()
            b.label(text="Warning: Zero distortion values.", icon='ERROR')
            layout.separator()
            
        b = layout.box()
        b.prop(props, "output_folder", text="")
        if not props.output_folder.strip():
            info_row = b.row()
            info_row.scale_y = 0.8
            info_row.label(text="* Saves next to .blend file when empty", icon='INFO')
            
        r = b.row()
        r.prop(props, "use_custom_name")
        if props.use_custom_name: r.prop(props, "custom_name")
        
        prefix = props.custom_name.strip() if props.use_custom_name and props.custom_name.strip() else get_clean_clip_name(clip.name)
        b.row().label(text=f"Preview: {prefix}_Undistort_STMap.{get_extension_from_format(props.file_format)}", icon='FILE_IMAGE')
        
        b = layout.box()
        b.label(text="Export Contents", icon='EXPORT')
        c = b.column(align=True)
        c.prop(props, "export_undistort")
        c.prop(props, "export_redistort")
        c.separator()
        c.prop(props, "export_grids")
        if props.export_grids: c.prop(props, "grid_count")
        c.separator()
        c.prop(props, "use_custom_resolution")
        if props.use_custom_resolution:
            r = c.row(align=True)
            r.prop(props, "custom_res_x"); r.prop(props, "custom_res_y")
            r.prop(props, "lock_aspect", icon='LOCKED' if props.lock_aspect else 'UNLOCKED', text="")
            r.operator("stmap.reset_custom_res", icon='FILE_REFRESH', text="")

        # ----------------------------------------------------
        # Overscan / BBox Panel
        # ----------------------------------------------------
        b = layout.box()
        row = b.row()
        row.alignment = 'LEFT'
        label_text = f"Overscan / BBox ({props.overscan_mode})"
        row.prop(props, "show_overscan", icon='TRIA_DOWN' if props.show_overscan else 'TRIA_RIGHT', text=label_text, emboss=False)
        
        if props.show_overscan:
            try:
                os_data = calculate_overscan_dimensions(context, props, clip)
                valid_data = True
            except Exception:
                valid_data = False

            b.prop(props, "overscan_mode", expand=True)
            
            if props.overscan_mode == 'CUSTOM':
                c = b.column(align=True)
                r = c.row(align=True)
                r.prop(props, "custom_overscan_percent")
                r.prop(props, "lock_overscan_aspect", icon='LOCKED' if props.lock_overscan_aspect else 'UNLOCKED', text="")
                
                r = c.row(align=True)
                r.prop(props, "custom_overscan_x")
                r.prop(props, "custom_overscan_y")
                
                if valid_data:
                    c.separator()
                    c.label(text=f"Auto Estimate Ref: {os_data['auto_w']} x {os_data['auto_h']}", icon='INFO')

            if valid_data:
                b.separator()
                
                # Res
                r = b.row()
                r.label(text=f"Res: {int(os_data['base_w'])}x{int(os_data['base_h'])}  ->  {os_data['out_w']}x{os_data['out_h']}")
                op = r.operator("stmap.copy_clipboard", text="", icon='COPYDOWN')
                op.text_to_copy = f"{os_data['out_w']}x{os_data['out_h']}"
                op.message = op.text_to_copy
                
                # Sensor
                r = b.row()
                r.label(text=f"Sensor: {os_data['sensor_width']:.2f}mm  ->  {os_data['os_sensor_width']:.2f}mm")
                op = r.operator("stmap.copy_clipboard", text="", icon='COPYDOWN')
                op.text_to_copy = f"{os_data['os_sensor_width']:.4f}"
                op.message = f"{os_data['os_sensor_width']:.2f}mm"
                
                # Target Camera
                scene_cam = context.scene.camera
                cam_name = scene_cam.name if scene_cam else "None"
                r = b.row()
                r.label(text=f"Target Camera: '{cam_name}'")
                
                b.separator()
                r = b.row(align=True)
                r.operator("stmap.apply_overscan", text="Apply to Scene", icon='CHECKMARK')
                r.operator("stmap.restore_overscan", text="Restore", icon='FILE_REFRESH')

        # ----------------------------------------------------
        # Options Panel
        # ----------------------------------------------------
        box = layout.box()
        row = box.row()
        row.alignment = 'LEFT'
        row.prop(props, "show_options", icon='TRIA_DOWN' if props.show_options else 'TRIA_RIGHT', text="Options", emboss=False)
        
        if props.show_options:
            col = box.column(align=True)
            
            col.separator()
            col.label(text="Performance:")
            perf_box = col.box()
            perf_box.prop(props, "use_gpu")
            
            if props.use_gpu and not GPU_AVAILABLE:
                perf_box.separator()
                perf_box.label(text="CuPy is not installed.", icon='INFO')
                perf_box.prop(props, "cuda_version")
                perf_box.operator("stmap.install_cupy", icon='IMPORT', text=f"Install CuPy ({props.cuda_version})")
                n = perf_box.column()
                n.scale_y = 0.8
                n.label(text="* Requires NVIDIA GPU", icon='ERROR')
                n.label(text="* Blender will freeze during install")
                
            col.separator()
            col.label(text="Format Settings:")
            fb = col.box()
            fb.prop(props, "file_format")
            if props.file_format == 'OPEN_EXR':
                r = fb.row(); r.prop(props, "exr_depth"); r.prop(props, "exr_codec")
            else: 
                fb.prop(props, "std_depth")
            fb.separator()
            fb.prop(props, "remap_bbox")
            
            col.separator()
            col.label(text="Auto BBox Limits:")
            ob = col.box()
            ob.prop(props, "max_overscan_percent")
            ob.prop(props, "extra_padding")

classes = (
    STMapExporterProperties,
    STMAP_OT_install_cupy,
    STMAP_OT_reset_custom_res,
    STMAP_OT_copy_clipboard,
    STMAP_OT_apply_overscan,
    STMAP_OT_restore_overscan,
    STMAP_OT_export,
    STMAP_MT_presets,
    STMAP_OT_add_preset,
    STMAP_PT_presets,
    STMAP_PT_main_panel,
)

def register():
    for cls in classes: bpy.utils.register_class(cls)
    bpy.types.Scene.stmap_exporter_props = bpy.props.PointerProperty(type=STMapExporterProperties)

def unregister():
    for cls in reversed(classes): bpy.utils.unregister_class(cls)
    del bpy.types.Scene.stmap_exporter_props
