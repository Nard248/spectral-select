"""
Wavelength Pipeline Helper Tool

Comprehensive tool for preparing data for the wavelength selection pipeline:
1. Visualize alignment between hyperspectral data and mask
2. Interactive ROI region drawing with coordinate output
3. Cropping configuration
4. Export pipeline-ready configuration

Usage:
    python wavelength_pipeline_helper.py
"""

import tkinter as tk
from tkinter import filedialog, ttk, messagebox
import numpy as np
from PIL import Image, ImageTk, ImageDraw
import pickle
import json
import os
from pathlib import Path


class WavelengthPipelineHelper:
    """Comprehensive helper for wavelength selection pipeline preparation."""

    # ROI Colors for different classes
    ROI_COLORS = [
        ('#FF0000', 'red'),
        ('#00FF00', 'green'),
        ('#0000FF', 'blue'),
        ('#FFFF00', 'yellow'),
        ('#FF00FF', 'magenta'),
        ('#00FFFF', 'cyan'),
        ('#FFA500', 'orange'),
        ('#800080', 'purple'),
    ]

    def __init__(self, root):
        """Initialize the helper tool."""
        self.root = root
        self.root.title("Wavelength Pipeline Helper")
        self.root.geometry("1600x950")

        # Data storage
        self.hyperspectral_data = None
        self.hyperspectral_visualization = None
        self.mask_image = None
        self.class_mask = None
        self.rgb_image = None

        # Spatial dimensions
        self.data_height = 0
        self.data_width = 0

        # ROI regions
        self.roi_regions = []
        self.current_roi_start = None
        self.drawing_roi = False

        # Cropping
        self.crop_x_start = tk.IntVar(value=0)
        self.crop_x_end = tk.IntVar(value=0)
        self.crop_y_start = tk.IntVar(value=0)
        self.crop_y_end = tk.IntVar(value=0)
        self.crop_enabled = tk.BooleanVar(value=False)

        # Display
        self.zoom_factor = 1.0
        self.display_mode = tk.StringVar(value="hyperspectral")
        self.overlay_mask = tk.BooleanVar(value=True)
        self.show_rois = tk.BooleanVar(value=True)

        # Create UI
        self.create_widgets()
        self.bind_events()

    def create_widgets(self):
        """Create the application widgets."""
        # Main container
        main_frame = ttk.Frame(self.root)
        main_frame.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)

        # Left panel - Controls
        control_panel = ttk.Frame(main_frame, width=350)
        control_panel.pack(side=tk.LEFT, fill=tk.Y, padx=5, pady=5)
        control_panel.pack_propagate(False)

        # === File Loading Section ===
        file_frame = ttk.LabelFrame(control_panel, text="1. Load Data")
        file_frame.pack(fill=tk.X, padx=5, pady=5)

        ttk.Button(file_frame, text="Load Hyperspectral PKL",
                   command=self.load_hyperspectral).pack(fill=tk.X, padx=5, pady=2)
        ttk.Button(file_frame, text="Load Mask/Labeled PNG",
                   command=self.load_mask).pack(fill=tk.X, padx=5, pady=2)
        ttk.Button(file_frame, text="Load RGB Image (optional)",
                   command=self.load_rgb).pack(fill=tk.X, padx=5, pady=2)

        self.data_info_label = ttk.Label(file_frame, text="No data loaded", wraplength=320)
        self.data_info_label.pack(anchor=tk.W, padx=5, pady=5)

        # === Display Options ===
        display_frame = ttk.LabelFrame(control_panel, text="2. Display Options")
        display_frame.pack(fill=tk.X, padx=5, pady=5)

        ttk.Label(display_frame, text="View Mode:").pack(anchor=tk.W, padx=5, pady=2)
        modes = [("Hyperspectral (Band Sum)", "hyperspectral"),
                 ("Mask/Labels", "mask"),
                 ("RGB Image", "rgb"),
                 ("Overlay (Data + Mask)", "overlay")]
        for text, mode in modes:
            ttk.Radiobutton(display_frame, text=text, variable=self.display_mode,
                            value=mode, command=self.update_display).pack(anchor=tk.W, padx=15)

        ttk.Checkbutton(display_frame, text="Show ROI regions",
                        variable=self.show_rois,
                        command=self.update_display).pack(anchor=tk.W, padx=5, pady=2)

        # === Cropping Section ===
        crop_frame = ttk.LabelFrame(control_panel, text="3. Cropping (Optional)")
        crop_frame.pack(fill=tk.X, padx=5, pady=5)

        ttk.Checkbutton(crop_frame, text="Enable Cropping",
                        variable=self.crop_enabled,
                        command=self.update_display).pack(anchor=tk.W, padx=5, pady=2)

        crop_grid = ttk.Frame(crop_frame)
        crop_grid.pack(fill=tk.X, padx=5, pady=2)

        ttk.Label(crop_grid, text="X:").grid(row=0, column=0, padx=2)
        ttk.Entry(crop_grid, textvariable=self.crop_x_start, width=6).grid(row=0, column=1, padx=2)
        ttk.Label(crop_grid, text="to").grid(row=0, column=2, padx=2)
        ttk.Entry(crop_grid, textvariable=self.crop_x_end, width=6).grid(row=0, column=3, padx=2)

        ttk.Label(crop_grid, text="Y:").grid(row=1, column=0, padx=2)
        ttk.Entry(crop_grid, textvariable=self.crop_y_start, width=6).grid(row=1, column=1, padx=2)
        ttk.Label(crop_grid, text="to").grid(row=1, column=2, padx=2)
        ttk.Entry(crop_grid, textvariable=self.crop_y_end, width=6).grid(row=1, column=3, padx=2)

        ttk.Button(crop_frame, text="Apply Crop Preview",
                   command=self.update_display).pack(fill=tk.X, padx=5, pady=2)

        # === ROI Region Section ===
        roi_frame = ttk.LabelFrame(control_panel, text="4. ROI Regions (Draw on Canvas)")
        roi_frame.pack(fill=tk.X, padx=5, pady=5)

        ttk.Label(roi_frame, text="Left-click + drag to draw ROI").pack(anchor=tk.W, padx=5, pady=2)

        self.roi_class_var = tk.IntVar(value=0)
        class_frame = ttk.Frame(roi_frame)
        class_frame.pack(fill=tk.X, padx=5, pady=2)
        ttk.Label(class_frame, text="Class:").pack(side=tk.LEFT)
        for i in range(5):
            color = self.ROI_COLORS[i][0]
            ttk.Radiobutton(class_frame, text=str(i),
                            variable=self.roi_class_var, value=i).pack(side=tk.LEFT, padx=2)

        roi_btn_frame = ttk.Frame(roi_frame)
        roi_btn_frame.pack(fill=tk.X, padx=5, pady=2)
        ttk.Button(roi_btn_frame, text="Clear Last",
                   command=self.clear_last_roi).pack(side=tk.LEFT, padx=2)
        ttk.Button(roi_btn_frame, text="Clear All",
                   command=self.clear_all_rois).pack(side=tk.LEFT, padx=2)
        ttk.Button(roi_btn_frame, text="Auto from Mask",
                   command=self.auto_rois_from_mask).pack(side=tk.LEFT, padx=2)

        # ROI List
        self.roi_listbox = tk.Listbox(roi_frame, height=6, font=('Courier', 9))
        self.roi_listbox.pack(fill=tk.X, padx=5, pady=5)

        # === Output Section ===
        output_frame = ttk.LabelFrame(control_panel, text="5. Export Configuration")
        output_frame.pack(fill=tk.X, padx=5, pady=5)

        ttk.Button(output_frame, text="Copy ROI Code to Clipboard",
                   command=self.copy_roi_code).pack(fill=tk.X, padx=5, pady=2)
        ttk.Button(output_frame, text="Export Pipeline Config JSON",
                   command=self.export_config).pack(fill=tk.X, padx=5, pady=2)
        ttk.Button(output_frame, text="Export Cropped Data",
                   command=self.export_cropped_data).pack(fill=tk.X, padx=5, pady=2)

        # === Code Preview ===
        code_frame = ttk.LabelFrame(control_panel, text="ROI Code Preview")
        code_frame.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)

        self.code_text = tk.Text(code_frame, height=10, width=40, font=('Courier', 9))
        self.code_text.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)

        # Right panel - Canvas
        canvas_frame = ttk.Frame(main_frame)
        canvas_frame.pack(side=tk.RIGHT, fill=tk.BOTH, expand=True, padx=5, pady=5)

        # Info bar
        self.coord_label = ttk.Label(canvas_frame, text="Position: (-, -) | Value: -")
        self.coord_label.pack(anchor=tk.W, padx=5, pady=2)

        # Canvas with scrollbars
        canvas_container = ttk.Frame(canvas_frame)
        canvas_container.pack(fill=tk.BOTH, expand=True)

        self.canvas = tk.Canvas(canvas_container, bg='gray20', cursor='crosshair')
        h_scroll = ttk.Scrollbar(canvas_container, orient=tk.HORIZONTAL, command=self.canvas.xview)
        v_scroll = ttk.Scrollbar(canvas_container, orient=tk.VERTICAL, command=self.canvas.yview)

        self.canvas.configure(xscrollcommand=h_scroll.set, yscrollcommand=v_scroll.set)

        h_scroll.pack(side=tk.BOTTOM, fill=tk.X)
        v_scroll.pack(side=tk.RIGHT, fill=tk.Y)
        self.canvas.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)

    def bind_events(self):
        """Bind mouse events."""
        self.canvas.bind('<Motion>', self.on_mouse_move)
        self.canvas.bind('<ButtonPress-1>', self.on_mouse_press)
        self.canvas.bind('<B1-Motion>', self.on_mouse_drag)
        self.canvas.bind('<ButtonRelease-1>', self.on_mouse_release)
        self.canvas.bind('<MouseWheel>', self.on_mouse_wheel)
        self.canvas.bind('<Button-4>', self.on_mouse_wheel)
        self.canvas.bind('<Button-5>', self.on_mouse_wheel)

    def load_hyperspectral(self):
        """Load hyperspectral data from PKL file."""
        file_path = filedialog.askopenfilename(
            title="Select Hyperspectral PKL",
            filetypes=[("Pickle Files", "*.pkl"), ("All Files", "*.*")]
        )
        if not file_path:
            return

        try:
            with open(file_path, 'rb') as f:
                self.hyperspectral_data = pickle.load(f)

            # Get dimensions from first excitation
            if 'data' in self.hyperspectral_data:
                first_ex = list(self.hyperspectral_data['data'].keys())[0]
                cube = self.hyperspectral_data['data'][first_ex]['cube']
                self.data_height, self.data_width = cube.shape[0], cube.shape[1]

                # Create visualization (sum across all bands and excitations)
                vis = np.zeros((self.data_height, self.data_width), dtype=np.float64)
                for ex_str, ex_data in self.hyperspectral_data['data'].items():
                    vis += np.sum(ex_data['cube'], axis=2)

                # Normalize for display
                vis = (vis - vis.min()) / (vis.max() - vis.min() + 1e-10) * 255
                self.hyperspectral_visualization = vis.astype(np.uint8)

                # Set crop defaults
                self.crop_x_end.set(self.data_width)
                self.crop_y_end.set(self.data_height)

                self.update_data_info()
                self.update_display()
                print(f"Loaded hyperspectral data: {self.data_height}x{self.data_width}")

        except Exception as e:
            messagebox.showerror("Error", f"Failed to load: {str(e)}")

    def load_mask(self):
        """Load mask or labeled PNG."""
        file_path = filedialog.askopenfilename(
            title="Select Mask/Labeled PNG",
            filetypes=[("Image Files", "*.png *.npy"), ("All Files", "*.*")]
        )
        if not file_path:
            return

        try:
            if file_path.endswith('.npy'):
                self.class_mask = np.load(file_path)
                # Create colored visualization
                self.mask_image = self.colorize_mask(self.class_mask)
            else:
                img = Image.open(file_path)
                if img.mode != 'RGB':
                    img = img.convert('RGB')
                self.mask_image = np.array(img)
                # Try to extract class mask
                self.class_mask = self.extract_class_mask(self.mask_image)

            self.update_data_info()
            self.update_display()
            print(f"Loaded mask: {self.mask_image.shape[:2]}")

        except Exception as e:
            messagebox.showerror("Error", f"Failed to load: {str(e)}")

    def load_rgb(self):
        """Load RGB image."""
        file_path = filedialog.askopenfilename(
            title="Select RGB Image",
            filetypes=[("Image Files", "*.jpg *.png *.bmp"), ("All Files", "*.*")]
        )
        if not file_path:
            return

        try:
            img = Image.open(file_path)
            if img.mode != 'RGB':
                img = img.convert('RGB')
            self.rgb_image = np.array(img)
            self.update_data_info()
            self.update_display()
            print(f"Loaded RGB: {self.rgb_image.shape[:2]}")

        except Exception as e:
            messagebox.showerror("Error", f"Failed to load: {str(e)}")

    def colorize_mask(self, mask):
        """Convert class mask to colored image."""
        h, w = mask.shape
        colored = np.zeros((h, w, 3), dtype=np.uint8)

        colors = [
            (255, 0, 0), (0, 255, 0), (0, 0, 255),
            (255, 255, 0), (255, 0, 255), (0, 255, 255)
        ]

        for class_id in range(6):
            class_mask = mask == class_id
            if np.any(class_mask):
                colored[class_mask] = colors[class_id % len(colors)]

        return colored

    def extract_class_mask(self, rgb_image):
        """Extract class mask from RGB image using color matching."""
        h, w = rgb_image.shape[:2]
        class_mask = np.full((h, w), -1, dtype=np.int16)

        class_colors = {
            (255, 0, 0): 0,
            (0, 255, 0): 1,
            (0, 0, 255): 2,
            (255, 255, 0): 3,
            (255, 0, 255): 4,
            (0, 255, 255): 5,
        }

        for color, class_id in class_colors.items():
            # Use tolerance for anti-aliasing
            mask = np.all(np.abs(rgb_image.astype(int) - np.array(color)) < 50, axis=2)
            class_mask[mask] = class_id

        return class_mask

    def update_data_info(self):
        """Update data info label."""
        info_parts = []

        if self.hyperspectral_data:
            info_parts.append(f"Hyperspectral: {self.data_height}x{self.data_width}")

        if self.mask_image is not None:
            info_parts.append(f"Mask: {self.mask_image.shape[0]}x{self.mask_image.shape[1]}")

        if self.rgb_image is not None:
            info_parts.append(f"RGB: {self.rgb_image.shape[0]}x{self.rgb_image.shape[1]}")

        # Check dimension match
        dims = []
        if self.hyperspectral_data:
            dims.append((self.data_height, self.data_width))
        if self.mask_image is not None:
            dims.append(self.mask_image.shape[:2])
        if self.rgb_image is not None:
            dims.append(self.rgb_image.shape[:2])

        if len(set(dims)) > 1:
            info_parts.append("\n⚠️ DIMENSION MISMATCH!")
        elif len(dims) > 1:
            info_parts.append("\n✓ Dimensions match")

        self.data_info_label.config(text="\n".join(info_parts))

    def update_display(self):
        """Update canvas display."""
        mode = self.display_mode.get()
        display_img = None

        if mode == "hyperspectral" and self.hyperspectral_visualization is not None:
            # Grayscale to RGB
            display_img = np.stack([self.hyperspectral_visualization] * 3, axis=2)

        elif mode == "mask" and self.mask_image is not None:
            display_img = self.mask_image.copy()

        elif mode == "rgb" and self.rgb_image is not None:
            display_img = self.rgb_image.copy()

        elif mode == "overlay":
            if self.hyperspectral_visualization is not None:
                base = np.stack([self.hyperspectral_visualization] * 3, axis=2).astype(float)
                if self.mask_image is not None:
                    # Resize mask if needed
                    mask = self.mask_image
                    if mask.shape[:2] != base.shape[:2]:
                        mask_pil = Image.fromarray(mask)
                        mask_pil = mask_pil.resize((base.shape[1], base.shape[0]), Image.Resampling.NEAREST)
                        mask = np.array(mask_pil)

                    # Create overlay where mask is non-black
                    mask_active = np.any(mask > 10, axis=2)
                    for c in range(3):
                        base[:, :, c] = np.where(
                            mask_active,
                            base[:, :, c] * 0.5 + mask[:, :, c] * 0.5,
                            base[:, :, c]
                        )
                display_img = base.astype(np.uint8)

        if display_img is None:
            return

        # Apply cropping preview
        if self.crop_enabled.get():
            x1, x2 = self.crop_x_start.get(), self.crop_x_end.get()
            y1, y2 = self.crop_y_start.get(), self.crop_y_end.get()

            # Draw crop rectangle
            display_img = display_img.copy()
            # Dim outside crop area
            dim_mask = np.ones(display_img.shape[:2], dtype=bool)
            dim_mask[y1:y2, x1:x2] = False
            display_img[dim_mask] = (display_img[dim_mask] * 0.3).astype(np.uint8)

        # Convert to PIL and draw ROIs
        pil_img = Image.fromarray(display_img)

        if self.show_rois.get():
            draw = ImageDraw.Draw(pil_img)
            for roi in self.roi_regions:
                color = self.ROI_COLORS[roi['class_id'] % len(self.ROI_COLORS)][0]
                y1, y2, x1, x2 = roi['coords']
                draw.rectangle([x1, y1, x2, y2], outline=color, width=2)
                draw.text((x1 + 2, y1 + 2), f"R{roi['id']}", fill=color)

        # Apply zoom
        if self.zoom_factor != 1.0:
            new_size = (int(pil_img.width * self.zoom_factor),
                        int(pil_img.height * self.zoom_factor))
            pil_img = pil_img.resize(new_size, Image.Resampling.NEAREST)

        self.photo = ImageTk.PhotoImage(pil_img)
        self.canvas.delete("all")
        self.canvas.create_image(0, 0, anchor=tk.NW, image=self.photo)
        self.canvas.configure(scrollregion=self.canvas.bbox("all"))

    def on_mouse_move(self, event):
        """Handle mouse movement for coordinate display."""
        x = int(self.canvas.canvasx(event.x) / self.zoom_factor)
        y = int(self.canvas.canvasy(event.y) / self.zoom_factor)

        value_str = "-"
        if self.hyperspectral_visualization is not None:
            h, w = self.hyperspectral_visualization.shape
            if 0 <= x < w and 0 <= y < h:
                value_str = f"{self.hyperspectral_visualization[y, x]}"

        self.coord_label.config(text=f"Position: ({x}, {y}) | Y={y}, X={x} | Value: {value_str}")

    def on_mouse_press(self, event):
        """Start drawing ROI."""
        x = int(self.canvas.canvasx(event.x) / self.zoom_factor)
        y = int(self.canvas.canvasy(event.y) / self.zoom_factor)
        self.current_roi_start = (x, y)
        self.drawing_roi = True

    def on_mouse_drag(self, event):
        """Update ROI preview while dragging."""
        if not self.drawing_roi or self.current_roi_start is None:
            return

        x = int(self.canvas.canvasx(event.x) / self.zoom_factor)
        y = int(self.canvas.canvasy(event.y) / self.zoom_factor)

        # Update coordinate display with ROI dimensions
        x1, y1 = self.current_roi_start
        w, h = abs(x - x1), abs(y - y1)
        self.coord_label.config(
            text=f"Drawing ROI: ({min(x1, x)}, {min(y1, y)}) to ({max(x1, x)}, {max(y1, y)}) | Size: {w}x{h}"
        )

    def on_mouse_release(self, event):
        """Finish drawing ROI."""
        if not self.drawing_roi or self.current_roi_start is None:
            return

        x = int(self.canvas.canvasx(event.x) / self.zoom_factor)
        y = int(self.canvas.canvasy(event.y) / self.zoom_factor)
        x1, y1 = self.current_roi_start

        # Ensure proper ordering
        x_start, x_end = min(x1, x), max(x1, x)
        y_start, y_end = min(y1, y), max(y1, y)

        # Minimum size check
        if (x_end - x_start) > 5 and (y_end - y_start) > 5:
            roi_id = len(self.roi_regions) + 1
            roi = {
                'id': roi_id,
                'name': f'Region {roi_id}',
                'coords': (y_start, y_end, x_start, x_end),  # (y_start, y_end, x_start, x_end)
                'class_id': self.roi_class_var.get(),
                'color': self.ROI_COLORS[self.roi_class_var.get() % len(self.ROI_COLORS)][0]
            }
            self.roi_regions.append(roi)
            self.update_roi_list()
            self.update_roi_code()
            self.update_display()

        self.drawing_roi = False
        self.current_roi_start = None

    def on_mouse_wheel(self, event):
        """Zoom with mouse wheel."""
        if event.num == 4 or event.delta > 0:
            self.zoom_factor *= 1.1
        else:
            self.zoom_factor *= 0.9
        self.zoom_factor = max(0.1, min(5.0, self.zoom_factor))
        self.update_display()

    def clear_last_roi(self):
        """Clear the last ROI."""
        if self.roi_regions:
            self.roi_regions.pop()
            self.update_roi_list()
            self.update_roi_code()
            self.update_display()

    def clear_all_rois(self):
        """Clear all ROIs."""
        self.roi_regions = []
        self.update_roi_list()
        self.update_roi_code()
        self.update_display()

    def auto_rois_from_mask(self):
        """Automatically generate ROI regions from class mask."""
        if self.class_mask is None:
            messagebox.showwarning("Warning", "No mask loaded!")
            return

        from scipy import ndimage

        self.roi_regions = []
        roi_id = 1

        for class_id in range(6):
            class_binary = (self.class_mask == class_id).astype(np.uint8)
            if not np.any(class_binary):
                continue

            # Find connected components
            labeled, num_objects = ndimage.label(class_binary)

            # Find largest object
            best_size = 0
            best_coords = None

            for obj_id in range(1, num_objects + 1):
                obj_mask = labeled == obj_id
                size = np.sum(obj_mask)

                if size > best_size:
                    best_size = size
                    ys, xs = np.where(obj_mask)
                    margin = 3
                    best_coords = (
                        max(0, int(ys.min()) - margin),
                        int(ys.max()) + margin,
                        max(0, int(xs.min()) - margin),
                        int(xs.max()) + margin
                    )

            if best_coords:
                roi = {
                    'id': roi_id,
                    'name': f'Region {roi_id}',
                    'coords': best_coords,
                    'class_id': class_id,
                    'color': self.ROI_COLORS[class_id % len(self.ROI_COLORS)][0]
                }
                self.roi_regions.append(roi)
                roi_id += 1

        self.update_roi_list()
        self.update_roi_code()
        self.update_display()
        messagebox.showinfo("Info", f"Generated {len(self.roi_regions)} ROI regions from mask")

    def update_roi_list(self):
        """Update ROI listbox."""
        self.roi_listbox.delete(0, tk.END)
        for roi in self.roi_regions:
            y1, y2, x1, x2 = roi['coords']
            self.roi_listbox.insert(
                tk.END,
                f"R{roi['id']}: ({y1},{y2},{x1},{x2}) C{roi['class_id']}"
            )

    def update_roi_code(self):
        """Update ROI code preview."""
        self.code_text.delete(1.0, tk.END)

        code = "ROI_REGIONS = [\n"
        for roi in self.roi_regions:
            y1, y2, x1, x2 = roi['coords']
            code += f"    {{'name': '{roi['name']}', 'coords': ({y1}, {y2}, {x1}, {x2}), 'color': '{roi['color']}'}},\n"
        code += "]\n"

        self.code_text.insert(tk.END, code)

    def copy_roi_code(self):
        """Copy ROI code to clipboard."""
        code = self.code_text.get(1.0, tk.END)
        self.root.clipboard_clear()
        self.root.clipboard_append(code)
        messagebox.showinfo("Copied", "ROI code copied to clipboard!")

    def export_config(self):
        """Export pipeline configuration as JSON."""
        file_path = filedialog.asksaveasfilename(
            title="Save Pipeline Config",
            defaultextension=".json",
            filetypes=[("JSON Files", "*.json"), ("All Files", "*.*")]
        )
        if not file_path:
            return

        config = {
            'dimensions': {
                'height': self.data_height,
                'width': self.data_width
            },
            'cropping': {
                'enabled': self.crop_enabled.get(),
                'x_start': self.crop_x_start.get(),
                'x_end': self.crop_x_end.get(),
                'y_start': self.crop_y_start.get(),
                'y_end': self.crop_y_end.get()
            },
            'roi_regions': [
                {
                    'name': roi['name'],
                    'coords': list(roi['coords']),
                    'color': roi['color'],
                    'class_id': roi['class_id']
                }
                for roi in self.roi_regions
            ]
        }

        with open(file_path, 'w') as f:
            json.dump(config, f, indent=2)

        messagebox.showinfo("Saved", f"Config saved to:\n{file_path}")

    def export_cropped_data(self):
        """Export cropped hyperspectral data and mask."""
        if self.hyperspectral_data is None:
            messagebox.showwarning("Warning", "No hyperspectral data loaded!")
            return

        if not self.crop_enabled.get():
            messagebox.showinfo("Info", "Cropping not enabled. Enable cropping first.")
            return

        folder = filedialog.askdirectory(title="Select Output Folder")
        if not folder:
            return

        try:
            x1, x2 = self.crop_x_start.get(), self.crop_x_end.get()
            y1, y2 = self.crop_y_start.get(), self.crop_y_end.get()

            # Crop hyperspectral data
            cropped_data = {
                'excitation_wavelengths': self.hyperspectral_data['excitation_wavelengths'],
                'metadata': self.hyperspectral_data.get('metadata', {}),
                'data': {}
            }

            for ex_str, ex_data in self.hyperspectral_data['data'].items():
                cropped_cube = ex_data['cube'][y1:y2, x1:x2, :]
                cropped_data['data'][ex_str] = {
                    **ex_data,
                    'cube': cropped_cube
                }

            # Save cropped data
            data_path = Path(folder) / "cropped_data.pkl"
            with open(data_path, 'wb') as f:
                pickle.dump(cropped_data, f)

            # Crop and save mask if available
            if self.class_mask is not None:
                cropped_mask = self.class_mask[y1:y2, x1:x2]
                mask_path = Path(folder) / "cropped_mask.npy"
                np.save(mask_path, cropped_mask)

                # Also save binary mask
                binary_mask = cropped_mask >= 0
                binary_path = Path(folder) / "cropped_binary_mask.npy"
                np.save(binary_path, binary_mask)

            messagebox.showinfo("Success",
                                f"Exported cropped data to:\n{folder}\n\n"
                                f"Cropped dimensions: {y2-y1}x{x2-x1}")

        except Exception as e:
            messagebox.showerror("Error", f"Export failed: {str(e)}")


def main():
    """Launch the helper tool."""
    root = tk.Tk()
    app = WavelengthPipelineHelper(root)
    root.mainloop()


if __name__ == "__main__":
    main()
