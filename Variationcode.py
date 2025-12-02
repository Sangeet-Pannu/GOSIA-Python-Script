import tkinter as tk
from tkinter import ttk, filedialog, messagebox
import os
import itertools
import datetime
import subprocess
import threading
import re
import time
import queue
import shutil  # Added for file copying

class GosiaScannerApp:
    def __init__(self, root): # constructor data type; runs when application begins; sets up the "brain" but does not operate it....
        self.root = root # Foundation of CODE
        self.root.title("GOSIA Advanced Scanner & Runner")
        self.root.geometry("950x950")

        # Data Storage
        self.filepath = ""
        self.raw_lines = []
        self.levels = {}        
        self.level_energies = {}
        self.transitions = []   
        self.modifications = {} 
        self.aux_files = []  # List to store filenames from OP,FILE
        self.gosia_path = ""
        
        # Multipolarity Map
        self.multipolarity_map = {
            '1': 'E1', '2': 'E2', '3': 'E3', '7': 'M1', 
            '8': 'E4', '9': 'M2'
        }
        
        self._setup_ui()

    def _setup_ui(self): # constructor data type; Sets up the GUI; acts as manager system;
        # Notebook for Tabs
        self.notebook = ttk.Notebook(self.root)
        self.notebook.pack(fill="both", expand=True, padx=5, pady=5)

        # --- TAB 1: EDITOR ---
        self.tab_editor = ttk.Frame(self.notebook)
        self.notebook.add(self.tab_editor, text="1. Edit & Generate")
        self._setup_editor_ui(self.tab_editor)

        # --- TAB 2: RUNNER ---
        self.tab_runner = ttk.Frame(self.notebook)
        self.notebook.add(self.tab_runner, text="2. Execute & Monitor")
        self._setup_runner_ui(self.tab_runner)

    def _setup_editor_ui(self, parent):
        # File Operations
        frame_file = ttk.LabelFrame(parent, text="File Operations", padding=10)
        frame_file.pack(fill="x", padx=10, pady=5)
        self.btn_load = ttk.Button(frame_file, text="Load .inp File", command=self.load_file)
        self.btn_load.pack(side="left", padx=5)
        self.lbl_status = ttk.Label(frame_file, text="No file loaded")
        self.lbl_status.pack(side="left", padx=5)

        # Selection
        frame_select = ttk.LabelFrame(parent, text="Select Transition", padding=10)
        frame_select.pack(fill="x", padx=10, pady=5)
        
        ttk.Label(frame_select, text="Select Level:").grid(row=0, column=0, sticky="w", pady=5)
        self.combo_levels = ttk.Combobox(frame_select, state="readonly", width=60)
        self.combo_levels.grid(row=0, column=1, sticky="ew", pady=5)
        self.combo_levels.bind("<<ComboboxSelected>>", self.update_transitions_menu)

        ttk.Label(frame_select, text="Select Transition:").grid(row=1, column=0, sticky="w", pady=5)
        self.combo_transitions = ttk.Combobox(frame_select, state="readonly", width=60)
        self.combo_transitions.grid(row=1, column=1, sticky="ew", pady=5)
        self.combo_transitions.bind("<<ComboboxSelected>>", self.on_transition_select)

        # Parameters
        frame_edit = ttk.LabelFrame(parent, text="Define Scan/Fix Parameters", padding=10)
        frame_edit.pack(fill="x", padx=10, pady=5)

        ttk.Label(frame_edit, text="Original Value:").grid(row=0, column=0, padx=5, pady=5, sticky="e")
        self.entry_original = ttk.Entry(frame_edit, state="readonly")
        self.entry_original.grid(row=0, column=1, padx=5, pady=5)

        self.var_fix_mode = tk.BooleanVar()
        self.chk_fix = ttk.Checkbutton(frame_edit, text="FIX Mode (Single Value)", 
                                       variable=self.var_fix_mode, command=self.toggle_fix_mode)
        self.chk_fix.grid(row=0, column=2, padx=10, pady=5, sticky="w")

        ttk.Label(frame_edit, text="Scan Start / Fixed Value:").grid(row=1, column=0, padx=5, pady=5, sticky="e")
        self.entry_start = ttk.Entry(frame_edit)
        self.entry_start.grid(row=1, column=1, padx=5, pady=5)

        self.lbl_stop = ttk.Label(frame_edit, text="Scan Stop:")
        self.lbl_stop.grid(row=2, column=0, padx=5, pady=5, sticky="e")
        self.entry_stop = ttk.Entry(frame_edit)
        self.entry_stop.grid(row=2, column=1, padx=5, pady=5)

        self.lbl_steps = ttk.Label(frame_edit, text="Number of Steps:")
        self.lbl_steps.grid(row=3, column=0, padx=5, pady=5, sticky="e")
        self.entry_steps = ttk.Entry(frame_edit)
        self.entry_steps.grid(row=3, column=1, padx=5, pady=5)
        self.entry_steps.insert(0, "5")

        self.btn_add = ttk.Button(frame_edit, text="Add/Update to List", command=self.add_modification, state="disabled")
        self.btn_add.grid(row=4, column=1, pady=10, sticky="ew")

        # List
        frame_list = ttk.LabelFrame(parent, text="Pending Modifications List", padding=10)
        frame_list.pack(fill="both", expand=True, padx=10, pady=5)
        self.listbox = tk.Listbox(frame_list, height=6)
        self.listbox.pack(side="left", fill="both", expand=True, padx=(0, 5))
        self.listbox.bind('<<ListboxSelect>>', self.on_list_select)
        
        scrollbar = ttk.Scrollbar(frame_list, orient="vertical", command=self.listbox.yview)
        scrollbar.pack(side="right", fill="y")
        self.listbox.config(yscrollcommand=scrollbar.set)
        
        btn_remove = ttk.Button(frame_list, text="Remove Selected", command=self.remove_modification)
        btn_remove.pack(side="bottom", pady=5)

        # Generate Button
        self.btn_generate = ttk.Button(parent, text="GENERATE BATCH FILES", command=self.generate_batch)
        self.btn_generate.pack(pady=10, fill="x", padx=20)

    def _setup_runner_ui(self, parent):
        frame_config = ttk.LabelFrame(parent, text="Execution Configuration", padding=10)
        frame_config.pack(fill="x", padx=10, pady=5)

        ttk.Label(frame_config, text="GOSIA Executable:").grid(row=0, column=0, sticky="e", padx=5)
        self.entry_gosia_path = ttk.Entry(frame_config, width=40)
        self.entry_gosia_path.grid(row=0, column=1, padx=5)
        self.entry_gosia_path.insert(0, "./gosia") 
        btn_browse_exec = ttk.Button(frame_config, text="Browse", command=self.browse_exec)
        btn_browse_exec.grid(row=0, column=2, padx=5)

        ttk.Label(frame_config, text="Batch Directory:").grid(row=1, column=0, sticky="e", padx=5)
        self.entry_batch_dir = ttk.Entry(frame_config, width=40)
        self.entry_batch_dir.grid(row=1, column=1, padx=5)
        btn_browse_dir = ttk.Button(frame_config, text="Browse", command=self.browse_batch_dir)
        btn_browse_dir.grid(row=1, column=2, padx=5)

        frame_physics = ttk.LabelFrame(parent, text="Convergence Criteria", padding=10)
        frame_physics.pack(fill="x", padx=10, pady=5)
        
        ttk.Label(frame_physics, text="Target Chi2:").grid(row=0, column=0, padx=5)
        self.entry_target_chi = ttk.Entry(frame_physics, width=10)
        self.entry_target_chi.insert(0, "1.0")
        self.entry_target_chi.grid(row=0, column=1, padx=5)

        ttk.Label(frame_physics, text="Convergence Tolerance:").grid(row=0, column=2, padx=5)
        self.entry_tolerance = ttk.Entry(frame_physics, width=10)
        self.entry_tolerance.insert(0, "0.001")
        self.entry_tolerance.grid(row=0, column=3, padx=5)

        ttk.Label(frame_physics, text="Max Restarts:").grid(row=0, column=4, padx=5)
        self.entry_restarts = ttk.Entry(frame_physics, width=10)
        self.entry_restarts.insert(0, "5")
        self.entry_restarts.grid(row=0, column=5, padx=5)

        self.btn_run = ttk.Button(parent, text="START PARALLEL PROCESSING", command=self.start_processing, state="normal")
        self.btn_run.pack(pady=10, fill="x", padx=20)

        self.tree_monitor = ttk.Treeview(parent, columns=("File", "Status", "Last Chi2", "Message"), show="headings", height=15)
        self.tree_monitor.heading("File", text="File")
        self.tree_monitor.heading("Status", text="Status")
        self.tree_monitor.heading("Last Chi2", text="Current Chi2")
        self.tree_monitor.heading("Message", text="Details")
        self.tree_monitor.column("File", width=200)
        self.tree_monitor.column("Status", width=100)
        self.tree_monitor.column("Last Chi2", width=100)
        self.tree_monitor.pack(fill="both", expand=True, padx=10, pady=5)

    def toggle_fix_mode(self):
        if self.var_fix_mode.get():
            self.entry_stop.config(state="disabled")
            self.entry_steps.config(state="disabled")
        else:
            self.entry_stop.config(state="normal")
            self.entry_steps.config(state="normal")

    def load_file(self):
        path = filedialog.askopenfilename(filetypes=[("GOSIA Input", "*.inp"), ("All Files", "*.*")])
        if not path: return
        self.filepath = path
        self.lbl_status.config(text=f"Loaded: {path.split('/')[-1]}")
        with open(path, 'r') as f: self.raw_lines = f.readlines()
        self.parse_data()
        self.update_levels_menu()
        self.modifications = {}
        self.listbox.delete(0, tk.END)

    def parse_data(self):
        self.levels = {}
        self.level_energies = {}
        self.transitions = []
        self.aux_files = [] # Reset aux files
        
        parsing_levels = False
        parsing_me = False
        parsing_files = False # Flag for OP,FILE
        current_multipolarity = "Unknown"
        
        for idx, line in enumerate(self.raw_lines):
            stripped = line.strip()
            clean_line = stripped.replace(',', ' ')
            parts = clean_line.split('!')[0].split()
            
            if not parts: continue
            
            # Detect OP,FILE
            if "OP,FILE" in stripped:
                parsing_files = True
                parsing_levels = False
                parsing_me = False
                continue
                
            if "LEVE" in stripped and not parsing_levels:
                parsing_levels = True
                parsing_files = False
                continue
            
            if "ME" in stripped and not parsing_me and "OP,ME" not in stripped:
                parsing_me = True
                parsing_levels = False
                parsing_files = False
                continue
            
            # --- PARSE AUX FILES ---
            if parsing_files:
                # Stop if we hit another OP code
                if stripped.startswith("OP,"):
                    parsing_files = False
                # Stop if we hit LEVE or ME (safety check)
                elif "LEVE" in stripped or "ME" in stripped:
                    parsing_files = False
                else:
                    # Logic: Ignore lines that look like "3,3,1" or "12 3 1"
                    # Generally files have dots or chars, ID lines are numbers
                    is_id_line = False
                    if len(parts) >= 3:
                        try:
                            # Check if first 3 parts are integers
                            int(parts[0])
                            int(parts[1])
                            int(parts[2])
                            is_id_line = True
                        except ValueError:
                            pass
                    
                    if not is_id_line:
                        # It's a filename
                        self.aux_files.append(stripped)

            # --- PARSE LEVELS ---
            elif parsing_levels:
                if parts[:4] == ['0', '0', '0', '0']: 
                    parsing_levels = False
                else:
                    try:
                        l_idx = parts[0]
                        en_mev = float(parts[3])
                        self.levels[l_idx] = f"Idx: {l_idx} | Spin: {parts[2]} | E: {en_mev} MeV"
                        self.level_energies[l_idx] = en_mev
                    except: pass

            # --- PARSE ME ---
            elif parsing_me:
                if parts[:5] == ['0', '0', '0', '0', '0']:
                    parsing_me = False
                    continue
                if len(parts) >= 5 and parts[1:5] == ['0', '0', '0', '0']:
                    key = parts[0]
                    current_multipolarity = self.multipolarity_map.get(key, f"Type-{key}")
                    continue
                if len(parts) >= 3:
                    try:
                        float(parts[2]) 
                        self.transitions.append({
                            'line_index': idx,
                            'idx1': parts[0],
                            'idx2': parts[1],
                            'type': current_multipolarity,
                            'value': parts[2]
                        })
                    except ValueError: pass

    def update_levels_menu(self):
        try:
            sorted_levels = sorted(self.levels.values(), key=lambda x: int(x.split('|')[0].split(':')[1]))
        except:
            sorted_levels = list(self.levels.values())
        self.combo_levels['values'] = sorted_levels
        self.combo_levels.set("Select a Level")
        self.combo_transitions.set("")

    def update_transitions_menu(self, event):
        selected_str = self.combo_levels.get()
        if not selected_str: return
        try:
            selected_idx = selected_str.split('|')[0].split(':')[1].strip()
        except: return
        
        relevant = []
        for i, trans in enumerate(self.transitions):
            if trans['idx1'] == selected_idx or trans['idx2'] == selected_idx:
                other = trans['idx2'] if trans['idx1'] == selected_idx else trans['idx1']
                e1 = self.level_energies.get(selected_idx, 0.0)
                e2 = self.level_energies.get(other, 0.0)
                diff_kev = abs(e1 - e2) * 1000.0
                display = f"{trans['type']}: {selected_idx}-{other} ({diff_kev:.1f} keV)"
                relevant.append((display, i))
        
        self.trans_menu_map = {t[0]: t[1] for t in relevant}
        self.combo_transitions['values'] = list(self.trans_menu_map.keys())

    def on_transition_select(self, event):
        val = self.combo_transitions.get()
        if not val: return
        
        trans_idx = self.trans_menu_map[val]
        trans_data = self.transitions[trans_idx]
        self.current_trans_idx = trans_idx 
        
        self.entry_original.config(state="normal")
        self.entry_original.delete(0, tk.END)
        self.entry_original.insert(0, trans_data['value'])
        self.entry_original.config(state="readonly")
        self.btn_add.config(state="normal")
        
        line_idx = trans_data['line_index']
        if line_idx in self.modifications:
            mod = self.modifications[line_idx]
            self.entry_start.delete(0, tk.END)
            self.entry_start.insert(0, str(mod['start']))
            self.var_fix_mode.set(mod['fixed'])
            self.toggle_fix_mode()
            if not mod['fixed']:
                self.entry_stop.delete(0, tk.END)
                self.entry_stop.insert(0, str(mod['stop']))
                self.entry_steps.delete(0, tk.END)
                self.entry_steps.insert(0, str(mod['steps']))
        else:
            self.entry_start.delete(0, tk.END)
            self.entry_stop.delete(0, tk.END)
            self.var_fix_mode.set(False)
            self.toggle_fix_mode()

    def add_modification(self):
        if not hasattr(self, 'current_trans_idx'): return
        trans = self.transitions[self.current_trans_idx]
        line_idx = trans['line_index']
        start = self.entry_start.get().strip()
        is_fixed = self.var_fix_mode.get()
        
        if not start: return
        if is_fixed:
            stop, steps = start, 1
            desc = f"[FIXED] {self.combo_transitions.get()} -> {start}"
        else:
            stop = self.entry_stop.get().strip()
            steps = self.entry_steps.get().strip()
            if not stop or not steps: return
            desc = f"[SCAN] {self.combo_transitions.get()} -> {start} to {stop} ({steps})"

        self.modifications[line_idx] = {
            'start': float(start), 'stop': float(stop), 'steps': int(steps),
            'fixed': is_fixed, 'display': desc,
            'trans_idx': self.current_trans_idx, 'name': self.combo_transitions.get()
        }
        self.refresh_listbox()

    def refresh_listbox(self):
        self.listbox.delete(0, tk.END)
        self.listbox_map = [] 
        for line_idx, data in self.modifications.items():
            self.listbox.insert(tk.END, data['display'])
            self.listbox_map.append(line_idx)

    def on_list_select(self, event):
        if not self.listbox.curselection(): return
        line_idx = self.listbox_map[self.listbox.curselection()[0]]
        mod = self.modifications[line_idx]
        trans_idx = mod['trans_idx']
        for name, idx in self.trans_menu_map.items():
            if idx == trans_idx:
                self.combo_transitions.set(name)
                break
        self.current_trans_idx = trans_idx
        self.entry_start.delete(0, tk.END)
        self.entry_start.insert(0, str(mod['start']))
        self.var_fix_mode.set(mod['fixed'])
        self.toggle_fix_mode()
        if not mod['fixed']:
            self.entry_stop.delete(0, tk.END)
            self.entry_stop.insert(0, str(mod['stop']))
            self.entry_steps.delete(0, tk.END)
            self.entry_steps.insert(0, str(mod['steps']))
        self.btn_add.config(state="normal")

    def remove_modification(self):
        if not self.listbox.curselection(): return
        line_idx = self.listbox_map[self.listbox.curselection()[0]]
        del self.modifications[line_idx]
        self.refresh_listbox()

    # --- GENERATION LOGIC ---
    def generate_scan_values(self, start, stop, steps):
        if steps <= 1: return [start]
        step_size = (stop - start) / (steps - 1)
        return [start + i * step_size for i in range(steps)]

    def generate_batch(self):
        if not self.modifications:
            messagebox.showwarning("Warning", "No modifications added.")
            return
        if not self.transitions:
            messagebox.showerror("Error", "No transitions parsed.")
            return

        out_dir = filedialog.askdirectory(title="Select Output Folder")
        if not out_dir: return
        
        try:
            timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
            batch_dir = os.path.join(out_dir, f"scan_results_{timestamp}")
            os.makedirs(batch_dir, exist_ok=True)

            # --- COPY AUX FILES ---
            source_dir = os.path.dirname(self.filepath)
            for fname in self.aux_files:
                src_path = os.path.join(source_dir, fname)
                dst_path = os.path.join(batch_dir, fname)
                try:
                    if os.path.exists(src_path):
                        shutil.copy(src_path, dst_path)
                    else:
                        print(f"Warning: Aux file not found: {src_path}")
                except Exception as e:
                    print(f"Error copying {fname}: {e}")

            # --- GENERATION ---
            base_me_values = [float(t['value']) for t in self.transitions]
            scanning_mods = []
            fixed_mods = []
            for l_idx, mod in self.modifications.items():
                if mod['fixed']: fixed_mods.append((l_idx, mod))
                else: 
                    vals = self.generate_scan_values(mod['start'], mod['stop'], mod['steps'])
                    scanning_mods.append({'line_idx': l_idx, 'vals': vals, 'mod_data': mod})

            scan_lists = [m['vals'] for m in scanning_mods]
            combinations = list(itertools.product(*scan_lists))

            for combo_idx, combo_values in enumerate(combinations):
                current_me_values = list(base_me_values)
                for l_idx, mod in fixed_mods:
                    if mod['trans_idx'] < len(current_me_values):
                        current_me_values[mod['trans_idx']] = mod['start']
                for i, val in enumerate(combo_values):
                    trans_idx = scanning_mods[i]['mod_data']['trans_idx']
                    if trans_idx < len(current_me_values):
                        current_me_values[trans_idx] = val

                me_filename = f"scan_step_{combo_idx}.me"
                with open(os.path.join(batch_dir, me_filename), 'w') as f_me:
                    for val in current_me_values:
                        f_me.write(f"{val:.17E}\n")

                new_content = self.raw_lines[:]
                for l_idx, mod in fixed_mods:
                    trans = self.transitions[mod['trans_idx']]
                    new_content[l_idx] = self.format_line(l_idx, trans, mod['start'], 1, 1)
                for i, val in enumerate(combo_values):
                    scan_info = scanning_mods[i]
                    trans = self.transitions[scan_info['mod_data']['trans_idx']]
                    new_content[scan_info['line_idx']] = self.format_line(scan_info['line_idx'], trans, val, 1, 1)

                inp_filename = f"scan_step_{combo_idx}.inp"
                out_filename = f"scan_step_{combo_idx}.out"
                
                for line_num, line in enumerate(new_content):
                    clean_line = line.replace(',', ' ')
                    if "12 3 1" in clean_line or "12,3,1" in clean_line:
                        if line_num + 1 < len(new_content):
                            new_content[line_num + 1] = f"{me_filename}\n"
                    if "22 3 1" in clean_line or "22,3,1" in clean_line:
                        if line_num + 1 < len(new_content):
                            new_content[line_num + 1] = f"{out_filename}\n"

                with open(os.path.join(batch_dir, inp_filename), 'w') as f:
                    f.writelines(new_content)

            self.entry_batch_dir.delete(0, tk.END)
            self.entry_batch_dir.insert(0, batch_dir)
            messagebox.showinfo("Success", f"Generated {len(combinations)} files in {batch_dir}")
            self.notebook.select(self.tab_runner) 

        except Exception as e:
            messagebox.showerror("Generation Error", str(e))

    def format_line(self, line_idx, trans, val, low, high):
        original = self.raw_lines[line_idx]
        comment = ""
        if "!" in original: comment = " ! " + original.split("!", 1)[1].strip()
        return f"{trans['idx1']}   {trans['idx2']}     {val:.5f}       {low} {high}{comment}\n"

    # --- RUNNER LOGIC ---
    def browse_exec(self):
        path = filedialog.askopenfilename(title="Select GOSIA Executable")
        if path:
            self.entry_gosia_path.delete(0, tk.END)
            self.entry_gosia_path.insert(0, path)

    def browse_batch_dir(self):
        path = filedialog.askdirectory(title="Select Batch Directory")
        if path:
            self.entry_batch_dir.delete(0, tk.END)
            self.entry_batch_dir.insert(0, path)

    def start_processing(self):
        gosia_cmd = self.entry_gosia_path.get()
        batch_dir = self.entry_batch_dir.get()
        
        if not os.path.exists(batch_dir): return
        inp_files = [f for f in os.listdir(batch_dir) if f.endswith(".inp")]
        if not inp_files: return

        for item in self.tree_monitor.get_children(): self.tree_monitor.delete(item)
        self.monitor_map = {} 
        for f in inp_files:
            item_id = self.tree_monitor.insert("", "end", values=(f, "Pending", "-", "Waiting..."))
            self.monitor_map[f] = item_id

        self.run_config = {
            'target_chi': float(self.entry_target_chi.get()),
            'tolerance': float(self.entry_tolerance.get()),
            'max_restarts': int(self.entry_restarts.get()),
            'cmd': gosia_cmd,
            'dir': batch_dir
        }
        threading.Thread(target=self.process_queue, args=(inp_files,), daemon=True).start()

    def process_queue(self, files):
        import multiprocessing
        max_workers = multiprocessing.cpu_count()
        self.gui_queue = queue.Queue()
        from concurrent.futures import ThreadPoolExecutor
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            futures = {executor.submit(self.run_single_gosia, f): f for f in files}
        self.root.after(100, self.check_gui_queue)

    def check_gui_queue(self):
        try:
            while True:
                msg = self.gui_queue.get_nowait()
                fname, status, chi2, text = msg
                if fname in self.monitor_map:
                    self.tree_monitor.set(self.monitor_map[fname], "Status", status)
                    self.tree_monitor.set(self.monitor_map[fname], "Last Chi2", chi2)
                    self.tree_monitor.set(self.monitor_map[fname], "Message", text)
        except queue.Empty: pass
        self.root.after(500, self.check_gui_queue)

    def run_single_gosia(self, filename):
        config = self.run_config
        full_path = os.path.join(config['dir'], filename)
        restart_count = 0
        best_chi = 999999.9
        prev_chi = 999999.9
        
        while restart_count <= config['max_restarts']:
            with open(full_path, 'r') as stdin_file:
                try:
                    process = subprocess.Popen(
                        [config['cmd']], stdin=stdin_file,
                        stdout=subprocess.PIPE, stderr=subprocess.PIPE,
                        cwd=config['dir'], text=True
                    )
                    while True:
                        line = process.stdout.readline()
                        if not line and process.poll() is not None: break
                        if line:
                            match = re.search(r"\*\*\* CHISQ=\s+([0-9\.E\+\-]+)", line)
                            if match:
                                try:
                                    val = float(match.group(1))
                                    best_chi = val
                                    self.gui_queue.put((filename, "Running", f"{val:.4e}", f"Iter {restart_count}"))
                                except: pass
                    
                    if best_chi <= config['target_chi']:
                        self.gui_queue.put((filename, "SUCCESS", f"{best_chi:.4e}", "Target Reached"))
                        return
                    if abs(prev_chi - best_chi) < config['tolerance']:
                        self.gui_queue.put((filename, "Converged", f"{best_chi:.4e}", "Change < Tol"))
                        return
                    prev_chi = best_chi
                    restart_count += 1
                    if restart_count <= config['max_restarts']:
                        self.gui_queue.put((filename, "Retrying", f"{best_chi:.4e}", f"Restarting ({restart_count})"))
                        time.sleep(0.5) 
                    else:
                        self.gui_queue.put((filename, "Done", f"{best_chi:.4e}", "Max Retries"))
                        return
                except Exception as e:
                    self.gui_queue.put((filename, "ERROR", "-", str(e)))
                    return

if __name__ == "__main__":
    root = tk.Tk() 
    app = GosiaScannerApp(root)
    root.mainloop()