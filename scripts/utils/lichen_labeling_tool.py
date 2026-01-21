"""
Lichen Labeling Tool

Interactive tool for labeling lichen samples with class assignments.
Workflow:
1. Load RGB image
2. Automatic thresholding to separate lichens from background
3. Connected component analysis to identify individual objects
4. Click on objects to assign them to classes
5. Save labeled PNG and class mask

Usage:
    python lichen_labeling_tool.py
"""

import tkinter as tk
from tkinter import filedialog, ttk, messagebox, simpledialog, colorchooser
import numpy as np
from PIL import Image, ImageTk
import cv2
from scipy import ndimage
import os
from pathlib import Path


class LichenLabelingTool:
    """Interactive tool for labeling lichen samples with class assignments."""

    # Default class colors (RGB)
    DEFAULT_COLORS = [
        (255, 0, 0),      # Red
        (0, 255, 0),      # Green
        (0, 0, 255),      # Blue
        (255, 255, 0),    # Yellow
        (255, 0, 255),    # Magenta
        (0, 255, 255),    # Cyan
        (255, 128, 0),    # Orange
        (128, 0, 255),    # Purple
        (0, 128, 255),    # Light Blue
        (255, 0, 128),    # Pink
    ]

    def __init__(self, root):
        """Initialize the labeling tool."""
        self.root = root
        self.root.title("Lichen Labeling Tool")
        self.root.geometry("1400x900")

        # Image and mask data
        self.original_image = None
        self.display_image = None
        self.binary_mask = None
        self.labeled_objects = None
        self.num_objects = 0
        self.object_labels = {}  # object_id -> class_id
        self.image_path = None

        # Class definitions: {class_id: {'name': str, 'color': tuple}}
        self.classes = {
            0: {'name': 'Background', 'color': (0, 0, 0)},
            1: {'name': 'Class 1', 'color': self.DEFAULT_COLORS[0]},
            2: {'name': 'Class 2', 'color': self.DEFAULT_COLORS[1]},
            3: {'name': 'Class 3', 'color': self.DEFAULT_COLORS[2]},
        }
        self.next_class_id = 4
        self.selected_class = 1

        # Thresholding parameters
        self.threshold_value = tk.IntVar(value=30)
        self.min_object_size = tk.IntVar(value=100)
        self.fill_holes = tk.BooleanVar(value=True)
        self.closing_size = tk.IntVar(value=5)

        # Multi-selection mode
        self.multi_select_mode = tk.BooleanVar(value=False)
        self.selected_objects = set()  # Objects selected for batch assignment

        # Display parameters
        self.zoom_factor = 1.0
        self.pan_x = 0
        self.pan_y = 0

        # Create the UI
        self.create_widgets()
        self.bind_events()

    def create_widgets(self):
        """Create the application widgets."""
        # Main container
        self.main_frame = ttk.Frame(self.root)
        self.main_frame.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)

        # Left panel - Controls
        self.control_panel = ttk.Frame(self.main_frame, width=300)
        self.control_panel.pack(side=tk.LEFT, fill=tk.Y, padx=5, pady=5)
        self.control_panel.pack_propagate(False)

        # File operations
        file_frame = ttk.LabelFrame(self.control_panel, text="File Operations")
        file_frame.pack(fill=tk.X, padx=5, pady=5)

        ttk.Button(file_frame, text="Load Image", command=self.load_image).pack(fill=tk.X, padx=5, pady=2)
        ttk.Button(file_frame, text="Save Labeled PNG", command=self.save_labeled_png).pack(fill=tk.X, padx=5, pady=2)
        ttk.Button(file_frame, text="Save Class Mask (.npy)", command=self.save_class_mask).pack(fill=tk.X, padx=5, pady=2)
        ttk.Button(file_frame, text="Save Resized Mask", command=self.save_resized_mask).pack(fill=tk.X, padx=5, pady=2)

        # Thresholding controls
        threshold_frame = ttk.LabelFrame(self.control_panel, text="Thresholding")
        threshold_frame.pack(fill=tk.X, padx=5, pady=5)

        ttk.Label(threshold_frame, text="Threshold Value:").pack(anchor=tk.W, padx=5, pady=2)
        threshold_slider = ttk.Scale(threshold_frame, from_=0, to=255, variable=self.threshold_value,
                                     orient=tk.HORIZONTAL, command=self.on_threshold_change)
        threshold_slider.pack(fill=tk.X, padx=5, pady=2)

        self.threshold_label = ttk.Label(threshold_frame, text=f"Value: {self.threshold_value.get()}")
        self.threshold_label.pack(anchor=tk.W, padx=5, pady=2)

        ttk.Label(threshold_frame, text="Min Object Size (pixels):").pack(anchor=tk.W, padx=5, pady=2)
        min_size_slider = ttk.Scale(threshold_frame, from_=10, to=1000, variable=self.min_object_size,
                                    orient=tk.HORIZONTAL, command=self.on_threshold_change)
        min_size_slider.pack(fill=tk.X, padx=5, pady=2)

        self.min_size_label = ttk.Label(threshold_frame, text=f"Min Size: {self.min_object_size.get()}")
        self.min_size_label.pack(anchor=tk.W, padx=5, pady=2)

        # Fill holes option
        ttk.Checkbutton(threshold_frame, text="Fill holes inside objects",
                        variable=self.fill_holes).pack(anchor=tk.W, padx=5, pady=2)

        ttk.Label(threshold_frame, text="Closing kernel size:").pack(anchor=tk.W, padx=5, pady=2)
        closing_slider = ttk.Scale(threshold_frame, from_=1, to=20, variable=self.closing_size,
                                   orient=tk.HORIZONTAL, command=self.on_threshold_change)
        closing_slider.pack(fill=tk.X, padx=5, pady=2)

        self.closing_label = ttk.Label(threshold_frame, text=f"Kernel: {self.closing_size.get()}")
        self.closing_label.pack(anchor=tk.W, padx=5, pady=2)

        ttk.Button(threshold_frame, text="Apply Thresholding", command=self.apply_thresholding).pack(fill=tk.X, padx=5, pady=5)

        # Class management
        class_frame = ttk.LabelFrame(self.control_panel, text="Classes")
        class_frame.pack(fill=tk.X, padx=5, pady=5)

        # Class listbox
        self.class_listbox = tk.Listbox(class_frame, height=8, selectmode=tk.SINGLE)
        self.class_listbox.pack(fill=tk.X, padx=5, pady=5)
        self.class_listbox.bind('<<ListboxSelect>>', self.on_class_select)

        # Class buttons
        class_btn_frame = ttk.Frame(class_frame)
        class_btn_frame.pack(fill=tk.X, padx=5, pady=2)

        ttk.Button(class_btn_frame, text="Add", command=self.add_class).pack(side=tk.LEFT, padx=2)
        ttk.Button(class_btn_frame, text="Rename", command=self.rename_class).pack(side=tk.LEFT, padx=2)
        ttk.Button(class_btn_frame, text="Color", command=self.change_class_color).pack(side=tk.LEFT, padx=2)
        ttk.Button(class_btn_frame, text="Delete", command=self.delete_class).pack(side=tk.LEFT, padx=2)

        self.selected_class_label = ttk.Label(class_frame, text="Selected: Class 1", font=('Helvetica', 10, 'bold'))
        self.selected_class_label.pack(anchor=tk.W, padx=5, pady=5)

        # Multi-selection mode
        select_frame = ttk.LabelFrame(self.control_panel, text="Multi-Select Mode")
        select_frame.pack(fill=tk.X, padx=5, pady=5)

        ttk.Checkbutton(select_frame, text="Enable multi-select",
                        variable=self.multi_select_mode,
                        command=self.on_multi_select_toggle).pack(anchor=tk.W, padx=5, pady=2)

        self.selection_label = ttk.Label(select_frame, text="Selected: 0 objects")
        self.selection_label.pack(anchor=tk.W, padx=5, pady=2)

        select_btn_frame = ttk.Frame(select_frame)
        select_btn_frame.pack(fill=tk.X, padx=5, pady=2)

        ttk.Button(select_btn_frame, text="Assign to Class",
                   command=self.assign_selected_to_class).pack(side=tk.LEFT, padx=2)
        ttk.Button(select_btn_frame, text="Clear Selection",
                   command=self.clear_selection).pack(side=tk.LEFT, padx=2)
        ttk.Button(select_btn_frame, text="Select All",
                   command=self.select_all_objects).pack(side=tk.LEFT, padx=2)

        # Statistics
        stats_frame = ttk.LabelFrame(self.control_panel, text="Statistics")
        stats_frame.pack(fill=tk.X, padx=5, pady=5)

        self.stats_label = ttk.Label(stats_frame, text="No image loaded")
        self.stats_label.pack(anchor=tk.W, padx=5, pady=5)

        # Instructions
        instr_frame = ttk.LabelFrame(self.control_panel, text="Instructions")
        instr_frame.pack(fill=tk.X, padx=5, pady=5)

        instructions = """1. Load an RGB image
2. Adjust threshold & enable
   "Fill holes" if needed
3. Click 'Apply Thresholding'
4. Select a class from the list
5. Click on lichens to assign

Multi-Select Mode:
- Enable to select multiple
  objects, then click "Assign
  to Class" for batch labeling
- "Select All" picks unlabeled

Shortcuts:
- Left click: Assign/Select
- Right click: Clear label
- Scroll: Zoom in/out
- Middle drag: Pan"""

        ttk.Label(instr_frame, text=instructions, justify=tk.LEFT).pack(anchor=tk.W, padx=5, pady=5)

        # Right panel - Image display
        self.canvas_frame = ttk.Frame(self.main_frame)
        self.canvas_frame.pack(side=tk.RIGHT, fill=tk.BOTH, expand=True, padx=5, pady=5)

        # Canvas with scrollbars
        self.canvas = tk.Canvas(self.canvas_frame, bg='gray20', cursor='crosshair')
        self.h_scrollbar = ttk.Scrollbar(self.canvas_frame, orient=tk.HORIZONTAL, command=self.canvas.xview)
        self.v_scrollbar = ttk.Scrollbar(self.canvas_frame, orient=tk.VERTICAL, command=self.canvas.yview)

        self.canvas.configure(xscrollcommand=self.h_scrollbar.set, yscrollcommand=self.v_scrollbar.set)

        self.h_scrollbar.pack(side=tk.BOTTOM, fill=tk.X)
        self.v_scrollbar.pack(side=tk.RIGHT, fill=tk.Y)
        self.canvas.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)

        # Initialize class listbox
        self.update_class_listbox()

    def bind_events(self):
        """Bind mouse and keyboard events."""
        self.canvas.bind('<Button-1>', self.on_canvas_click)
        self.canvas.bind('<Button-3>', self.on_canvas_right_click)
        self.canvas.bind('<MouseWheel>', self.on_mouse_wheel)
        self.canvas.bind('<Button-4>', self.on_mouse_wheel)  # Linux scroll up
        self.canvas.bind('<Button-5>', self.on_mouse_wheel)  # Linux scroll down
        self.canvas.bind('<ButtonPress-2>', self.on_pan_start)
        self.canvas.bind('<B2-Motion>', self.on_pan_move)
        self.canvas.bind('<Motion>', self.on_mouse_move)

    def load_image(self):
        """Load an RGB image."""
        file_path = filedialog.askopenfilename(
            title="Select RGB Image",
            filetypes=[
                ("Image Files", "*.jpg *.jpeg *.png *.bmp *.tiff"),
                ("All Files", "*.*")
            ]
        )

        if not file_path:
            return

        try:
            # Load image
            self.image_path = file_path
            img = Image.open(file_path)

            # Convert to RGB if needed
            if img.mode != 'RGB':
                img = img.convert('RGB')

            self.original_image = np.array(img)
            self.display_image = self.original_image.copy()

            # Reset masks and labels
            self.binary_mask = None
            self.labeled_objects = None
            self.num_objects = 0
            self.object_labels = {}

            # Reset zoom and pan
            self.zoom_factor = 1.0
            self.pan_x = 0
            self.pan_y = 0

            # Update display
            self.update_display()
            self.update_stats()

            print(f"Loaded image: {os.path.basename(file_path)}")
            print(f"Image shape: {self.original_image.shape}")

        except Exception as e:
            messagebox.showerror("Error", f"Failed to load image: {str(e)}")

    def on_threshold_change(self, event=None):
        """Handle threshold slider changes."""
        self.threshold_label.config(text=f"Value: {self.threshold_value.get()}")
        self.min_size_label.config(text=f"Min Size: {self.min_object_size.get()}")
        self.closing_label.config(text=f"Kernel: {self.closing_size.get()}")

    def apply_thresholding(self):
        """Apply thresholding to separate objects from background."""
        if self.original_image is None:
            messagebox.showwarning("Warning", "Please load an image first!")
            return

        try:
            # Convert to grayscale
            gray = cv2.cvtColor(self.original_image, cv2.COLOR_RGB2GRAY)

            # Apply threshold
            threshold = self.threshold_value.get()
            _, binary = cv2.threshold(gray, threshold, 255, cv2.THRESH_BINARY)

            # Fill holes inside objects if enabled
            if self.fill_holes.get():
                # Morphological closing to connect nearby regions
                closing_size = self.closing_size.get()
                kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (closing_size, closing_size))
                binary = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel)

                # Fill holes using flood fill from edges
                # Create a slightly larger image for flood fill
                h, w = binary.shape
                flood_fill_mask = np.zeros((h + 2, w + 2), np.uint8)
                binary_copy = binary.copy()

                # Flood fill from the corner (assumes corner is background)
                cv2.floodFill(binary_copy, flood_fill_mask, (0, 0), 255)

                # Invert flood-filled image
                binary_inv = cv2.bitwise_not(binary_copy)

                # Combine with original to fill holes
                binary = binary | binary_inv

            # Remove small objects
            min_size = self.min_object_size.get()

            # Label connected components
            labeled, num_features = ndimage.label(binary)

            # Filter by size
            sizes = ndimage.sum(binary, labeled, range(1, num_features + 1))
            mask_sizes = sizes >= min_size

            # Create filtered binary mask
            filtered_mask = np.zeros_like(binary)
            for i, keep in enumerate(mask_sizes, 1):
                if keep:
                    filtered_mask[labeled == i] = 255

            # Re-label filtered objects
            self.labeled_objects, self.num_objects = ndimage.label(filtered_mask)
            self.binary_mask = filtered_mask > 0

            # Reset object labels and selection
            self.object_labels = {}
            self.selected_objects = set()
            self.update_selection_label()

            # Update display
            self.update_display()
            self.update_stats()

            print(f"Found {self.num_objects} objects after thresholding")

        except Exception as e:
            messagebox.showerror("Error", f"Thresholding failed: {str(e)}")

    def update_display(self):
        """Update the canvas display."""
        if self.original_image is None:
            return

        # Start with original image
        display = self.original_image.copy()

        # Overlay object boundaries if thresholding has been applied
        if self.labeled_objects is not None:
            # Find object boundaries
            boundaries = self.find_boundaries(self.labeled_objects)

            # Draw boundaries in white
            display[boundaries] = [255, 255, 255]

            # Color labeled objects
            for obj_id, class_id in self.object_labels.items():
                if class_id in self.classes:
                    color = self.classes[class_id]['color']
                    obj_mask = self.labeled_objects == obj_id

                    # Create colored overlay with transparency
                    for c in range(3):
                        display[:, :, c] = np.where(
                            obj_mask,
                            np.clip(display[:, :, c] * 0.4 + color[c] * 0.6, 0, 255).astype(np.uint8),
                            display[:, :, c]
                        )

            # Highlight selected objects (in multi-select mode) with cyan border
            if self.multi_select_mode.get() and self.selected_objects:
                for obj_id in self.selected_objects:
                    obj_mask = self.labeled_objects == obj_id
                    # Draw thicker cyan boundary around selected objects
                    kernel = np.ones((5, 5), dtype=np.uint8)
                    dilated = cv2.dilate(obj_mask.astype(np.uint8), kernel, iterations=2)
                    selection_boundary = dilated.astype(bool) & ~obj_mask
                    display[selection_boundary] = [0, 255, 255]  # Cyan

        self.display_image = display

        # Convert to PhotoImage and display
        img_pil = Image.fromarray(display)

        # Apply zoom
        if self.zoom_factor != 1.0:
            new_size = (int(img_pil.width * self.zoom_factor), int(img_pil.height * self.zoom_factor))
            img_pil = img_pil.resize(new_size, Image.Resampling.NEAREST)

        self.photo = ImageTk.PhotoImage(img_pil)

        # Update canvas
        self.canvas.delete("all")
        self.canvas.create_image(0, 0, anchor=tk.NW, image=self.photo)
        self.canvas.configure(scrollregion=self.canvas.bbox("all"))

    def find_boundaries(self, labeled_array):
        """Find boundaries between labeled objects."""
        # Use morphological gradient to find edges
        kernel = np.ones((3, 3), dtype=np.uint8)
        dilated = cv2.dilate(labeled_array.astype(np.uint8), kernel, iterations=1)
        eroded = cv2.erode(labeled_array.astype(np.uint8), kernel, iterations=1)
        boundaries = dilated != eroded
        return boundaries

    def update_stats(self):
        """Update the statistics display."""
        if self.original_image is None:
            self.stats_label.config(text="No image loaded")
            return

        h, w = self.original_image.shape[:2]
        stats = f"Image: {w} x {h}\n"

        if self.num_objects > 0:
            labeled_count = len(self.object_labels)
            stats += f"Objects: {self.num_objects}\n"
            stats += f"Labeled: {labeled_count}\n"
            stats += f"Unlabeled: {self.num_objects - labeled_count}"

            # Class breakdown
            class_counts = {}
            for obj_id, class_id in self.object_labels.items():
                class_counts[class_id] = class_counts.get(class_id, 0) + 1

            if class_counts:
                stats += "\n\nBy class:"
                for class_id, count in sorted(class_counts.items()):
                    if class_id in self.classes:
                        stats += f"\n  {self.classes[class_id]['name']}: {count}"

        self.stats_label.config(text=stats)

    def update_class_listbox(self):
        """Update the class listbox."""
        self.class_listbox.delete(0, tk.END)

        for class_id in sorted(self.classes.keys()):
            if class_id == 0:
                continue  # Skip background class
            class_info = self.classes[class_id]
            color = class_info['color']
            self.class_listbox.insert(tk.END, f"{class_info['name']} (RGB: {color[0]},{color[1]},{color[2]})")

        # Select first class
        if self.class_listbox.size() > 0:
            self.class_listbox.selection_set(0)
            self.on_class_select(None)

    def on_class_select(self, event):
        """Handle class selection from listbox."""
        selection = self.class_listbox.curselection()
        if selection:
            idx = selection[0]
            # Get class_id (skip background at index 0)
            class_ids = sorted([k for k in self.classes.keys() if k != 0])
            if idx < len(class_ids):
                self.selected_class = class_ids[idx]
                class_name = self.classes[self.selected_class]['name']
                color = self.classes[self.selected_class]['color']
                self.selected_class_label.config(
                    text=f"Selected: {class_name}",
                    foreground=f'#{color[0]:02x}{color[1]:02x}{color[2]:02x}'
                )

    def add_class(self):
        """Add a new class."""
        name = simpledialog.askstring("Add Class", "Enter class name:")
        if name:
            # Assign next available color
            color_idx = (self.next_class_id - 1) % len(self.DEFAULT_COLORS)
            color = self.DEFAULT_COLORS[color_idx]

            self.classes[self.next_class_id] = {'name': name, 'color': color}
            self.next_class_id += 1
            self.update_class_listbox()

    def rename_class(self):
        """Rename the selected class."""
        if self.selected_class == 0:
            messagebox.showwarning("Warning", "Cannot rename background class!")
            return

        current_name = self.classes[self.selected_class]['name']
        new_name = simpledialog.askstring("Rename Class", "Enter new name:", initialvalue=current_name)

        if new_name:
            self.classes[self.selected_class]['name'] = new_name
            self.update_class_listbox()

    def change_class_color(self):
        """Change the color of the selected class."""
        if self.selected_class == 0:
            messagebox.showwarning("Warning", "Cannot change background color!")
            return

        current_color = self.classes[self.selected_class]['color']
        color = colorchooser.askcolor(
            initialcolor=f'#{current_color[0]:02x}{current_color[1]:02x}{current_color[2]:02x}',
            title="Choose Class Color"
        )

        if color[0]:
            self.classes[self.selected_class]['color'] = tuple(int(c) for c in color[0])
            self.update_class_listbox()
            self.update_display()

    def delete_class(self):
        """Delete the selected class."""
        if self.selected_class == 0:
            messagebox.showwarning("Warning", "Cannot delete background class!")
            return

        if self.selected_class in [1, 2, 3]:
            if not messagebox.askyesno("Confirm", f"Delete {self.classes[self.selected_class]['name']}?"):
                return

        # Remove class and its labels
        del self.classes[self.selected_class]
        self.object_labels = {k: v for k, v in self.object_labels.items() if v != self.selected_class}

        self.update_class_listbox()
        self.update_display()
        self.update_stats()

    def on_canvas_click(self, event):
        """Handle left click on canvas to assign class or select object."""
        if self.labeled_objects is None:
            return

        # Convert canvas coordinates to image coordinates
        x = int(self.canvas.canvasx(event.x) / self.zoom_factor)
        y = int(self.canvas.canvasy(event.y) / self.zoom_factor)

        # Check bounds
        h, w = self.labeled_objects.shape
        if 0 <= x < w and 0 <= y < h:
            obj_id = self.labeled_objects[y, x]

            if obj_id > 0:
                if self.multi_select_mode.get():
                    # Toggle object selection
                    if obj_id in self.selected_objects:
                        self.selected_objects.remove(obj_id)
                        print(f"Deselected object {obj_id}")
                    else:
                        self.selected_objects.add(obj_id)
                        print(f"Selected object {obj_id}")
                    self.update_selection_label()
                else:
                    # Direct assignment mode - assign selected class to this object
                    self.object_labels[obj_id] = self.selected_class
                    print(f"Object {obj_id} -> {self.classes[self.selected_class]['name']}")
                    self.update_stats()
                self.update_display()

    def on_canvas_right_click(self, event):
        """Handle right click to clear label."""
        if self.labeled_objects is None:
            return

        # Convert canvas coordinates to image coordinates
        x = int(self.canvas.canvasx(event.x) / self.zoom_factor)
        y = int(self.canvas.canvasy(event.y) / self.zoom_factor)

        # Check bounds
        h, w = self.labeled_objects.shape
        if 0 <= x < w and 0 <= y < h:
            obj_id = self.labeled_objects[y, x]

            if obj_id > 0 and obj_id in self.object_labels:
                del self.object_labels[obj_id]
                print(f"Object {obj_id} label cleared")
                self.update_display()
                self.update_stats()

    def on_mouse_wheel(self, event):
        """Handle mouse wheel for zooming."""
        # Get scroll direction
        if event.num == 4 or event.delta > 0:
            factor = 1.1
        else:
            factor = 0.9

        self.zoom_factor *= factor
        self.zoom_factor = max(0.1, min(10.0, self.zoom_factor))

        self.update_display()

    def on_pan_start(self, event):
        """Start panning."""
        self.canvas.scan_mark(event.x, event.y)

    def on_pan_move(self, event):
        """Pan the canvas."""
        self.canvas.scan_dragto(event.x, event.y, gain=1)

    def on_mouse_move(self, event):
        """Show object info on mouse hover."""
        if self.labeled_objects is None:
            return

        # Convert canvas coordinates to image coordinates
        x = int(self.canvas.canvasx(event.x) / self.zoom_factor)
        y = int(self.canvas.canvasy(event.y) / self.zoom_factor)

        # Check bounds
        h, w = self.labeled_objects.shape
        if 0 <= x < w and 0 <= y < h:
            obj_id = self.labeled_objects[y, x]

            if obj_id > 0:
                if obj_id in self.object_labels:
                    class_id = self.object_labels[obj_id]
                    class_name = self.classes.get(class_id, {}).get('name', 'Unknown')
                    self.root.title(f"Lichen Labeling Tool - Object {obj_id}: {class_name}")
                else:
                    self.root.title(f"Lichen Labeling Tool - Object {obj_id}: Unlabeled")
            else:
                self.root.title("Lichen Labeling Tool - Background")
        else:
            self.root.title("Lichen Labeling Tool")

    # --- Multi-selection methods ---

    def on_multi_select_toggle(self):
        """Handle toggling of multi-select mode."""
        if not self.multi_select_mode.get():
            # Exiting multi-select mode - clear selection
            self.selected_objects = set()
        self.update_selection_label()
        self.update_display()

    def update_selection_label(self):
        """Update the selection count label."""
        count = len(self.selected_objects)
        self.selection_label.config(text=f"Selected: {count} objects")

    def assign_selected_to_class(self):
        """Assign all selected objects to the current class."""
        if not self.selected_objects:
            messagebox.showinfo("Info", "No objects selected. Click on objects to select them first.")
            return

        count = len(self.selected_objects)
        class_name = self.classes[self.selected_class]['name']

        for obj_id in self.selected_objects:
            self.object_labels[obj_id] = self.selected_class

        print(f"Assigned {count} objects to {class_name}")

        # Clear selection after assignment
        self.selected_objects = set()
        self.update_selection_label()
        self.update_display()
        self.update_stats()

        messagebox.showinfo("Success", f"Assigned {count} objects to {class_name}")

    def clear_selection(self):
        """Clear all selected objects."""
        self.selected_objects = set()
        self.update_selection_label()
        self.update_display()

    def select_all_objects(self):
        """Select all unlabeled objects."""
        if self.labeled_objects is None:
            return

        # Select all objects that don't have a label yet
        for obj_id in range(1, self.num_objects + 1):
            if obj_id not in self.object_labels:
                self.selected_objects.add(obj_id)

        self.update_selection_label()
        self.update_display()
        print(f"Selected {len(self.selected_objects)} unlabeled objects")

    def save_labeled_png(self):
        """Save the labeled image as PNG."""
        if self.original_image is None or self.labeled_objects is None:
            messagebox.showwarning("Warning", "No labeled data to save!")
            return

        file_path = filedialog.asksaveasfilename(
            title="Save Labeled PNG",
            defaultextension=".png",
            filetypes=[("PNG Files", "*.png"), ("All Files", "*.*")],
            initialdir=os.path.dirname(self.image_path) if self.image_path else None,
            initialfile="labeled_lichens.png"
        )

        if not file_path:
            return

        try:
            # Create labeled image
            h, w = self.labeled_objects.shape
            labeled_img = np.zeros((h, w, 3), dtype=np.uint8)

            # Fill background with black
            labeled_img[:] = self.classes[0]['color']

            # Fill each labeled object with its class color
            for obj_id, class_id in self.object_labels.items():
                if class_id in self.classes:
                    color = self.classes[class_id]['color']
                    obj_mask = self.labeled_objects == obj_id
                    labeled_img[obj_mask] = color

            # Fill unlabeled objects with gray
            for obj_id in range(1, self.num_objects + 1):
                if obj_id not in self.object_labels:
                    obj_mask = self.labeled_objects == obj_id
                    labeled_img[obj_mask] = (128, 128, 128)  # Gray for unlabeled

            # Save
            img = Image.fromarray(labeled_img)
            img.save(file_path)

            print(f"Saved labeled PNG to: {file_path}")
            messagebox.showinfo("Success", f"Labeled PNG saved to:\n{file_path}")

        except Exception as e:
            messagebox.showerror("Error", f"Failed to save: {str(e)}")

    def save_class_mask(self):
        """Save the class mask as numpy array."""
        if self.original_image is None or self.labeled_objects is None:
            messagebox.showwarning("Warning", "No labeled data to save!")
            return

        file_path = filedialog.asksaveasfilename(
            title="Save Class Mask",
            defaultextension=".npy",
            filetypes=[("NumPy Files", "*.npy"), ("All Files", "*.*")],
            initialdir=os.path.dirname(self.image_path) if self.image_path else None,
            initialfile="class_mask.npy"
        )

        if not file_path:
            return

        try:
            # Create class mask
            h, w = self.labeled_objects.shape
            class_mask = np.zeros((h, w), dtype=np.uint8)

            # Assign class IDs to each pixel
            for obj_id, class_id in self.object_labels.items():
                obj_mask = self.labeled_objects == obj_id
                class_mask[obj_mask] = class_id

            # Save
            np.save(file_path, class_mask)

            # Also save class definitions
            class_def_path = file_path.replace('.npy', '_classes.txt')
            with open(class_def_path, 'w') as f:
                f.write("# Class definitions: class_id, name, R, G, B\n")
                for class_id, info in sorted(self.classes.items()):
                    color = info['color']
                    f.write(f"{class_id}, {info['name']}, {color[0]}, {color[1]}, {color[2]}\n")

            print(f"Saved class mask to: {file_path}")
            print(f"Saved class definitions to: {class_def_path}")
            messagebox.showinfo("Success",
                                f"Class mask saved to:\n{file_path}\n\nClass definitions saved to:\n{class_def_path}")

        except Exception as e:
            messagebox.showerror("Error", f"Failed to save: {str(e)}")

    def save_resized_mask(self):
        """Save the class mask resized to match hyperspectral data dimensions."""
        if self.original_image is None or self.labeled_objects is None:
            messagebox.showwarning("Warning", "No labeled data to save!")
            return

        # Ask for target dimensions
        target_dims = simpledialog.askstring(
            "Target Dimensions",
            "Enter target dimensions (height,width):\n(e.g., 256,348 for Lichens_2 data)",
            initialvalue="256,348"
        )

        if not target_dims:
            return

        try:
            target_h, target_w = map(int, target_dims.split(','))
        except ValueError:
            messagebox.showerror("Error", "Invalid dimensions. Use format: height,width")
            return

        file_path = filedialog.asksaveasfilename(
            title="Save Resized Class Mask",
            defaultextension=".npy",
            filetypes=[("NumPy Files", "*.npy"), ("All Files", "*.*")],
            initialdir=os.path.dirname(self.image_path) if self.image_path else None,
            initialfile=f"class_mask_{target_h}x{target_w}.npy"
        )

        if not file_path:
            return

        try:
            # Create class mask at original resolution
            h, w = self.labeled_objects.shape
            class_mask = np.zeros((h, w), dtype=np.uint8)

            # Assign class IDs to each pixel
            for obj_id, class_id in self.object_labels.items():
                obj_mask = self.labeled_objects == obj_id
                class_mask[obj_mask] = class_id

            # Resize using nearest neighbor to preserve class labels
            resized_mask = cv2.resize(
                class_mask,
                (target_w, target_h),
                interpolation=cv2.INTER_NEAREST
            )

            # Save
            np.save(file_path, resized_mask)

            # Also save class definitions
            class_def_path = file_path.replace('.npy', '_classes.txt')
            with open(class_def_path, 'w') as f:
                f.write("# Class definitions: class_id, name, R, G, B\n")
                f.write(f"# Resized from {h}x{w} to {target_h}x{target_w}\n")
                for class_id, info in sorted(self.classes.items()):
                    color = info['color']
                    f.write(f"{class_id}, {info['name']}, {color[0]}, {color[1]}, {color[2]}\n")

            print(f"Saved resized mask ({target_h}x{target_w}) to: {file_path}")
            messagebox.showinfo("Success",
                                f"Resized mask ({target_h}x{target_w}) saved to:\n{file_path}")

        except Exception as e:
            messagebox.showerror("Error", f"Failed to save: {str(e)}")


def main():
    """Launch the lichen labeling tool."""
    root = tk.Tk()
    app = LichenLabelingTool(root)
    root.mainloop()


if __name__ == "__main__":
    main()
