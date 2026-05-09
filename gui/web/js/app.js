document.addEventListener('DOMContentLoaded', () => {
    
    // Tab Switching Logic
    const navBtns = document.querySelectorAll('.nav-btn');
    const tabContents = document.querySelectorAll('.tab-content');

    navBtns.forEach(btn => {
        btn.addEventListener('click', () => {
            // Remove active from all
            navBtns.forEach(b => b.classList.remove('active'));
            tabContents.forEach(tc => {
                tc.classList.remove('active');
                tc.classList.add('hidden');
            });

            // Add active to clicked
            btn.classList.add('active');
            const targetId = btn.getAttribute('data-tab');
            const targetTab = document.getElementById(targetId);
            targetTab.classList.remove('hidden');
            // Little delay to allow display block before opacity animation
            setTimeout(() => targetTab.classList.add('active'), 10);
            
            // If help tab is opened for the first time, load README
            if(targetId === 'help-tab' && !window.readmeLoaded) {
                loadReadme();
            }
            
            // Re-render plotly charts if container became visible to fix sizing issues
            if(targetId === 'plots-tab' && window.plotsAvailable) {
                window.dispatchEvent(new Event('resize'));
            }
        });
    });

    // Run Optimization Button
    const btnRun = document.getElementById('btn_run');
    const logsContainer = document.getElementById('logs_container');
    const terminalOutput = document.getElementById('terminal_output');
    const statusIndicator = document.getElementById('status_indicator');

    btnRun.addEventListener('click', () => {
        // Gather params
        const params = {
            project_path: document.getElementById('project_path').value,
            fitness_fn_name: document.getElementById('fitness_fn_name').value,
            num_generations: parseInt(document.getElementById('num_generations').value),
            population_size: parseInt(document.getElementById('population_size').value),
            elite_count: parseInt(document.getElementById('elite_count').value),
            selection: document.getElementById('selection').value,
            crossover: document.getElementById('crossover').value,
            mutation: document.getElementById('mutation').value,
            min_values: document.getElementById('min_values').value,
            max_values: document.getElementById('max_values').value
        };

        if(!params.project_path || !params.fitness_fn_name) {
            alert("Please provide the project path and fitness function name.");
            return;
        }

        // Setup UI for running
        logsContainer.classList.remove('hidden');
        terminalOutput.innerHTML = '';
        statusIndicator.innerHTML = '<i class="fas fa-circle-notch fa-spin mr-2"></i> Running...';
        statusIndicator.className = 'flex items-center text-sm font-medium text-yellow-400';
        btnRun.disabled = true;
        btnRun.innerHTML = '<i class="fas fa-spinner fa-spin mr-2"></i> Optimizing...';
        btnRun.classList.add('opacity-50', 'cursor-not-allowed');

        // Call python function
        eel.run_optimization(params)();
    });

    // Browse Button
    const btnBrowse = document.getElementById('btn_browse');
    if(btnBrowse) {
        btnBrowse.addEventListener('click', async () => {
            try {
                const path = await eel.ask_file()();
                if (path) {
                    document.getElementById('project_path').value = path;
                }
            } catch(e) {
                console.error("Browse error:", e);
            }
        });
    }

    // Export Button
    const btnExport = document.getElementById('btn_export_xlsx');
    if(btnExport) {
        btnExport.addEventListener('click', async () => {
            btnExport.innerHTML = '<i class="fas fa-spinner fa-spin mr-2"></i> Exporting...';
            const success = await eel.export_data()();
            if(success) {
                btnExport.innerHTML = '<i class="fas fa-check mr-2"></i> Exported!';
                setTimeout(() => {
                    btnExport.innerHTML = '<i class="fas fa-download mr-2"></i> Export Data';
                }, 3000);
            } else {
                btnExport.innerHTML = '<i class="fas fa-times mr-2"></i> Failed';
                setTimeout(() => {
                    btnExport.innerHTML = '<i class="fas fa-download mr-2"></i> Export Data';
                }, 3000);
            }
        });
    }

    // Load Button
    const btnLoad = document.getElementById('btn_load_xlsx');
    if(btnLoad) {
        btnLoad.addEventListener('click', async () => {
            const params = {
                min_values: document.getElementById('min_values').value,
                max_values: document.getElementById('max_values').value
            };
            btnLoad.innerHTML = '<i class="fas fa-spinner fa-spin mr-2"></i> Loading...';
            try {
                const result = await eel.load_xlsx(params)();
                if(result && result.success) {
                    optimization_finished(true, {plots: result.plots});
                } else if(result && result.error) {
                    alert("Error loading file: " + result.error);
                }
            } catch(e) {
                console.error("Load error:", e);
            } finally {
                btnLoad.innerHTML = '<i class="fas fa-upload mr-2"></i> Load .xlsx';
            }
        });
    }
});

// Exposed function to receive log lines from Python
eel.expose(update_log);
function update_log(message, level) {
    const terminal = document.getElementById('terminal_output');
    if(!terminal) return; // Prevent KeyError if DOM is not ready
    
    const div = document.createElement('div');
    
    // Simple color coding based on level/message content
    if(level === 'ERROR') {
        div.className = 'text-red-400';
    } else if(level === 'WARNING') {
        div.className = 'text-yellow-400 font-bold';
    } else if (message && message.includes('Best Fitness')) {
        div.className = 'text-blue-300';
    } else {
        div.className = 'text-green-400';
    }
    
    div.textContent = message;
    terminal.appendChild(div);
    
    // Auto scroll to bottom
    terminal.scrollTop = terminal.scrollHeight;
}

// Exposed function when optimization finishes
eel.expose(optimization_finished);
function optimization_finished(success, result_data) {
    const btnRun = document.getElementById('btn_run');
    const statusIndicator = document.getElementById('status_indicator');
    
    btnRun.disabled = false;
    btnRun.innerHTML = '<i class="fas fa-play mr-2"></i> Run Optimization';
    btnRun.classList.remove('opacity-50', 'cursor-not-allowed');

    if(success) {
        statusIndicator.innerHTML = '<i class="fas fa-check-circle mr-2"></i> Completed';
        statusIndicator.className = 'flex items-center text-sm font-medium text-green-400';
        
        // Handle Plots
        if(result_data && result_data.plots) {
            document.getElementById('plots_placeholder').classList.add('hidden');
            document.getElementById('plots_container').classList.remove('hidden');
            document.getElementById('btn_export_xlsx').classList.remove('hidden');
            
            window.plotsAvailable = true;
            
            const layoutConfig = {
                responsive: true,
                toImageButtonOptions: {
                    format: 'png', // one of png, svg, jpeg, webp
                    filename: 'plot_export',
                    height: 800,
                    width: 1200,
                    scale: 2,
                    setBackground: 'transparent'
                }
            };
            
            // Plotly render
            if(result_data.plots.fit_plot) {
                const fig = JSON.parse(result_data.plots.fit_plot);
                Plotly.newPlot('plotly_fitplot', fig.data, fig.layout, layoutConfig);
            }
            if(result_data.plots.parallel_plot) {
                const fig = JSON.parse(result_data.plots.parallel_plot);
                Plotly.newPlot('plotly_parallel', fig.data, fig.layout, layoutConfig);
            }
            if(result_data.plots.box_plot) {
                const fig = JSON.parse(result_data.plots.box_plot);
                Plotly.newPlot('plotly_boxplot', fig.data, fig.layout, layoutConfig);
            }
            
            // Auto switch to plots tab
            document.querySelector('[data-tab="plots-tab"]').click();
        }
    } else {
        statusIndicator.innerHTML = '<i class="fas fa-times-circle mr-2"></i> Error';
        statusIndicator.className = 'flex items-center text-sm font-medium text-red-400';
        update_log(result_data.error || "Unknown error occurred.", "ERROR");
    }
}

async function loadReadme() {
    try {
        const md_text = await eel.get_readme_content()();
        if(md_text) {
            document.getElementById('readme_content').innerHTML = marked.parse(md_text);
        } else {
            document.getElementById('readme_content').innerHTML = "<p>Could not load README.md.</p>";
        }
        window.readmeLoaded = true;
    } catch(e) {
        console.error("Error loading README:", e);
        document.getElementById('readme_content').innerHTML = "<p>Error loading README.</p>";
    }
}
