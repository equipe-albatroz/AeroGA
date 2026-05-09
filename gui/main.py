import os
import sys
import threading
import logging
import importlib.util
import json
import eel
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import tkinter as tk
from tkinter import filedialog

# Add src to path so AeroGA can be imported
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'src')))
from AeroGA.AeroGA import optimize
import AeroGA.settings as settings

# Initialize Eel
eel.init(os.path.join(os.path.dirname(__file__), 'web'))

@eel.expose
def ask_file():
    root = tk.Tk()
    root.withdraw()
    root.wm_attributes('-topmost', 1)
    file_path = filedialog.askopenfilename(filetypes=[("Python Files", "*.py")])
    return file_path

class EelLogHandler(logging.Handler):
    def emit(self, record):
        msg = record.getMessage()
        level = record.levelname
        try:
            eel.update_log(msg, level)()
        except Exception:
            pass

# Add our handler to the AeroGA logger
eel_handler = EelLogHandler()
eel_handler.setFormatter(logging.Formatter('%(message)s'))
settings.log.addHandler(eel_handler)

def generate_plotly_json(out, num_generations, min_values, max_values):
    plots = {}
    history = out["history_valid"]
    values_gen = out["values_gen"]
    
    # 1. Fit Plot
    fig_fit = go.Figure()
    fig_fit.add_trace(go.Scatter(y=values_gen["best_fit"], mode='lines', name='Best Fit', line=dict(color='#38bdf8', width=3)))
    
    # avg_fit might have None values initially
    avg_fit_clean = [x if x is not None else values_gen["best_fit"][i] for i, x in enumerate(values_gen["avg_fit"])]
    fig_fit.add_trace(go.Scatter(y=avg_fit_clean, mode='lines', name='Avg Fit', line=dict(color='#fb7185', dash='dash')))
    
    if "metrics" in values_gen and values_gen["metrics"]:
        fig_fit.add_trace(go.Scatter(y=values_gen["metrics"], mode='lines', name='Diversity', yaxis='y2', line=dict(color='#a78bfa')))
    
    fig_fit.update_layout(
        title="Fitness & Diversity Evolution",
        xaxis_title="Generations",
        yaxis_title="Fitness",
        yaxis2=dict(title="Diversity", overlaying='y', side='right'),
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(0,0,0,0)',
        font=dict(color='#cbd5e1'),
        margin=dict(l=40, r=40, t=40, b=40)
    )
    fig_fit.update_xaxes(showgrid=True, gridwidth=1, gridcolor='rgba(148, 163, 184, 0.8)', griddash='dot')
    fig_fit.update_yaxes(showgrid=True, gridwidth=1, gridcolor='rgba(148, 163, 184, 0.2)')
    
    plots['fit_plot'] = fig_fit.to_json()

    # 2. Boxplot
    try:
        num_ind = len(history["ind"])
        if num_ind > 0:
            num_var = len(history["ind"][0])
            data_aux = []
            for i in range(num_ind):
                row = []
                for j in range(num_var):
                    if max_values[j] - min_values[j] != 0:
                        row.append((history["ind"][i][j] - min_values[j]) / (max_values[j] - min_values[j]))
                    else:
                        row.append(0)
                data_aux.append(row)
            
            df_box = pd.DataFrame(data_aux, columns=[f'Var_{i+1}' for i in range(num_var)])
            fig_box = go.Figure()
            for col in df_box.columns:
                fig_box.add_trace(go.Box(y=df_box[col], name=col))
            
            fig_box.update_layout(
                title='Dispersion of Normalized Values',
                paper_bgcolor='rgba(0,0,0,0)',
                plot_bgcolor='rgba(0,0,0,0)',
                font=dict(color='#cbd5e1'),
                margin=dict(l=40, r=40, t=40, b=40)
            )
            plots['box_plot'] = fig_box.to_json()

            # 3. Parallel Coordinates
            df_par = pd.DataFrame(history["ind"], columns=[f'Var_{i+1}' for i in range(num_var)])
            df_par['Score'] = history["score"]
            
            fig_par = px.parallel_coordinates(df_par, color="Score", dimensions=df_par.columns, 
                                              color_continuous_scale=px.colors.diverging.Tealrose)
            
            fig_par.update_layout(
                title="Parallel Coordinates",
                paper_bgcolor='rgba(0,0,0,0)',
                plot_bgcolor='rgba(0,0,0,0)',
                font=dict(color='#cbd5e1'),
                margin=dict(l=40, r=40, t=40, b=40)
            )
            plots['parallel_plot'] = fig_par.to_json()
    except Exception as e:
        settings.log.error(f"Error generating plots: {e}")

    return plots

@eel.expose
def run_optimization(params):
    def optimization_thread():
        try:
            # Parse parameters
            project_path = params['project_path']
            fitness_fn_name = params['fitness_fn_name']
            
            min_vals = [float(x.strip()) for x in params['min_values'].split(',')]
            max_vals = [float(x.strip()) for x in params['max_values'].split(',')]
            
            if len(min_vals) != len(max_vals):
                raise ValueError("Length of min_values and max_values must be equal.")
            
            # Dynamically import module
            module_name = "user_fitness_module"
            spec = importlib.util.spec_from_file_location(module_name, project_path)
            if spec is None:
                raise ImportError(f"Could not load module from {project_path}")
            
            user_module = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(user_module)
            
            if not hasattr(user_module, fitness_fn_name):
                raise AttributeError(f"Function {fitness_fn_name} not found in {project_path}")
            
            fitness_fn = getattr(user_module, fitness_fn_name)
            
            settings.log.info(f"Starting Optimization via UI using {fitness_fn_name}...")
            
            out = optimize(
                selection=params['selection'],
                crossover=params['crossover'],
                mutation=params['mutation'],
                n_threads=-1,
                min_values=min_vals,
                max_values=max_vals,
                population_size=params['population_size'],
                num_generations=params['num_generations'],
                elite_count=params['elite_count'],
                fitness_fn=fitness_fn,
                plotfit=False, plotbox=False, plotparallel=False, report=False, export_xlsx=False
            )
            
            if out:
                global current_out
                current_out = out
                settings.log.info("Generating interactive plots...")
                plots = generate_plotly_json(out, params['num_generations'], min_vals, max_vals)
                eel.optimization_finished(True, {"plots": plots})()
            else:
                eel.optimization_finished(False, {"error": "Optimization returned None."})()
                
        except Exception as e:
            import traceback
            settings.log.error(f"Error during optimization: {str(e)}")
            settings.log.error(traceback.format_exc())
            eel.optimization_finished(False, {"error": str(e)})()

    # Run in thread
    t = threading.Thread(target=optimization_thread)
    t.start()

current_out = None

@eel.expose
def export_data():
    global current_out
    if not current_out: return False
    
    root = tk.Tk()
    root.withdraw()
    root.wm_attributes('-topmost', 1)
    file_path = filedialog.asksaveasfilename(defaultextension=".xlsx", filetypes=[("Excel files", "*.xlsx")])
    if not file_path: return False
    
    try:
        history = current_out["history_valid"]
        lista = list(history["score"])
        data = [list(x) for x in zip(*history["ind"])]
        df = pd.DataFrame(data).transpose()
        num_var = len(history["ind"][0])
        df.columns = [f'Var_{i+1}' for i in range(num_var)]
        df['fit'] = list(history["fit"])
        df['Score'] = lista
        df['gen'] = list(history["gen"])
        df.to_excel(file_path, index=False)
        return True
    except Exception as e:
        settings.log.error(f"Export error: {e}")
        return False

@eel.expose
def load_xlsx(params):
    try:
        min_vals = [float(x.strip()) for x in params['min_values'].split(',')] if params.get('min_values') else []
        max_vals = [float(x.strip()) for x in params['max_values'].split(',')] if params.get('max_values') else []
        
        root = tk.Tk()
        root.withdraw()
        root.wm_attributes('-topmost', 1)
        file_path = filedialog.askopenfilename(filetypes=[("Excel files", "*.xlsx")])
        if not file_path: return {"error": "No file selected."}
        
        df = pd.read_excel(file_path)
        
        # Build a mock 'out' object to use with generate_plotly_json
        var_cols = [c for c in df.columns if c.startswith('Var_')]
        if not var_cols:
            return {"error": "File missing Var_X columns."}
            
        history_valid = {
            "ind": df[var_cols].values.tolist(),
            "score": df['Score'].tolist(),
            "fit": df['fit'].tolist(),
            "gen": df['gen'].tolist()
        }
        
        # Calculate mock values_gen (best_fit and avg_fit)
        best_fit_list = []
        avg_fit_list = []
        generations = sorted(df['gen'].unique())
        for g in generations:
            gen_data = df[df['gen'] == g]
            best_fit_list.append(gen_data['fit'].min())
            avg_fit_list.append(gen_data['fit'].mean())
            
        values_gen = {
            "best_fit": best_fit_list,
            "avg_fit": avg_fit_list,
            "metrics": [] # We don't have diversity saved in xlsx
        }
        
        out = {"history_valid": history_valid, "values_gen": values_gen}
        
        if len(min_vals) != len(var_cols):
            min_vals = [0] * len(var_cols)
            max_vals = [0] * len(var_cols)
            
        plots = generate_plotly_json(out, len(generations), min_vals, max_vals)
        
        # Save as current_out so we can export it again if needed
        global current_out
        current_out = out
        
        return {"success": True, "plots": plots}
    except Exception as e:
        settings.log.error(f"Load error: {e}")
        return {"error": str(e)}

@eel.expose
def get_readme_content():
    readme_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'README.md'))
    if os.path.exists(readme_path):
        with open(readme_path, 'r', encoding='utf-8') as f:
            return f.read()
    return ""

if __name__ == '__main__':
    # Add Eel block for compiling
    eel.start('index.html', size=(1200, 800), port=0)
