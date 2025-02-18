# frontend/app.py

from flask import Flask, render_template, request, redirect, url_for, send_from_directory, flash, jsonify, Response
import subprocess
import os
import json
import plotly
import sys

# Add parent directory to the system path to import functions from main.py
sys.path.append(os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))

from plot import (
    create_tree_plot, 
    create_aggregation_plot, 
    create_treemap_plot, 
    create_sunburst_plot,
    create_subtree_plot,
    node_to_dict,
    load_tree_json,
    create_score_distribution_plot,
    get_labels_parallel,
    create_spider_chart
)

app = Flask(__name__)
app.secret_key = 'your_secret_key'  # Replace with a secure key

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
JSON_FOLDER = os.path.join(BASE_DIR, '..', 'json')
MAIN_PY_PATH = os.path.join(BASE_DIR, '..', 'main.py')
PLOTS_FOLDER = os.path.join(BASE_DIR, '..', 'plots')
SPIDER_FOLDER = os.path.join(BASE_DIR, '..', 'spider')

@app.route('/', methods=['GET', 'POST'])
def index():
    if request.method == 'POST':
        params = {
            'YEAR_OLD_START': request.form.get('YEAR_OLD_START'),
            'YEAR_OLD_END': request.form.get('YEAR_OLD_END'),
            'YEAR_NEW_START': request.form.get('YEAR_NEW_START'),
            'YEAR_NEW_END': request.form.get('YEAR_NEW_END'),
            'GENERATION_TYPE': request.form.get('GENERATION_TYPE'),
            'SIMILARITY_TYPE': request.form.get('SIMILARITY_TYPE'),
            'NUM_SAMPLES': request.form.get('NUM_SAMPLES'),
            'SUBCATEGORY_COUNT': request.form.get('SUBCATEGORY_COUNT'),
            'TOKENS_PER_FACT': request.form.get('TOKENS_PER_FACT'),
            'OUTPUT_FILENAME': request.form.get('OUTPUT_FILENAME'),
            'ITERATION_DEPTH': request.form.get('ITERATION_DEPTH'),
            'ROOT_TOPIC_NAME': request.form.get('ROOT_TOPIC_NAME'),
            'THRESHOLD_SIMILARITY_CLASSES': request.form.get('THRESHOLD_SIMILARITY_CLASSES'),
            'THRESHOLD_SKIP_NODE_HIGH': request.form.get('THRESHOLD_SKIP_NODE_HIGH'),
            'THRESHOLD_SKIP_NODE_LOW': request.form.get('THRESHOLD_SKIP_NODE_LOW'),
        }
        print(f"Running model with parameters: {params}")
        try:
            cmd = ['python', MAIN_PY_PATH]
            for key, value in params.items():
                cmd.extend([f'--{key}', str(value)])
            subprocess.run(cmd, check=True)
            flash('Algorithm executed successfully!', 'success')
        except subprocess.CalledProcessError as e:
            flash(f'An error occurred: {e}', 'danger')

        return redirect(url_for('index'))

    json_files = []
    if os.path.exists(JSON_FOLDER):
        json_files = [f for f in os.listdir(JSON_FOLDER) if f.endswith('.json')]

    plot_files = []
    if os.path.exists(PLOTS_FOLDER):
        for root, dirs, files in os.walk(PLOTS_FOLDER):
            rel_path = os.path.relpath(root, PLOTS_FOLDER)
            for file in files:
                if file.endswith('.html'):
                    file_path = os.path.join(rel_path, file) if rel_path != '.' else file
                    plot_files.append(file_path)

    return render_template('index.html', json_files=json_files, plot_files=plot_files)

@app.route('/json/<filename>')
def get_json_file(filename):
    return send_from_directory(JSON_FOLDER, filename)

@app.route('/plots/<path:filename>')
def serve_plot(filename):
    safe_path = os.path.normpath(filename)
    if '..' in safe_path.split(os.sep):
        return jsonify({'error': 'Invalid file path.'}), 400

    plots_dir = os.path.abspath(PLOTS_FOLDER)
    file_path = os.path.join(plots_dir, safe_path)
    if not os.path.exists(file_path):
        return jsonify({'error': 'File does not exist.'}), 404

    return send_from_directory(plots_dir, safe_path)

@app.route('/plot', methods=['POST'])
def plot():
    data = request.get_json()
    filename = data.get('filename')
    plot_type = data.get('plot_type')
    visible_traces = data.get('visible_traces', ['new','old','diff'])  # Default to all if not provided

    if not filename or not plot_type:
        return jsonify({'error': 'Filename and plot type are required.'}), 400

    filepath = os.path.join(JSON_FOLDER, filename)
    if not os.path.exists(filepath):
        return jsonify({'error': 'File does not exist.'}), 400

    # Handle Spider plot
    if plot_type == 'spider':
        spider_cache_file_path = os.path.join(SPIDER_FOLDER, filename)
        spider_cache_file = os.path.splitext(spider_cache_file_path)[0] + "_spider_data.json"

        if os.path.exists(spider_cache_file):
            # Load from cache
            with open(spider_cache_file, 'r') as f:
                spider_data = json.load(f)
            label_counts_new = spider_data['label_counts_new']
            label_counts_old = spider_data['label_counts_old']
            labels = spider_data['labels']
        else:
            label_counts_new, label_counts_old, labels = get_labels_parallel(filepath)
            spider_data = {
                'label_counts_new': label_counts_new,
                'label_counts_old': label_counts_old,
                'labels': labels
            }
            with open(spider_cache_file, 'w') as f:
                json.dump(spider_data, f)

        visibility = {trace: (trace in visible_traces) for trace in ['new','old','diff']}
        fig_spider = create_spider_chart(label_counts_new, label_counts_old, labels, visibility)
        distribution_plot_json = create_score_distribution_plot(filepath)
        response = {
            'selected_plot': fig_spider,
            'score_distribution_plot': distribution_plot_json
        }
        fig_serialized = json.dumps(response, cls=plotly.utils.PlotlyJSONEncoder)
        return Response(fig_serialized, mimetype='application/json')

    # Other plot types
    plot_functions = {
        'tree': create_tree_plot,
        'tree_node_aggregation': create_aggregation_plot,
        'treemap': create_treemap_plot,
        'sunburst': create_sunburst_plot,
        'subtree': create_subtree_plot
    }

    plot_func = plot_functions.get(plot_type)
    if not plot_func:
        return jsonify({'error': 'Invalid plot type.'}), 400

    try:
        selected_plot_json = plot_func(filepath)
        distribution_plot_json = create_score_distribution_plot(filepath)
        response = {
            'selected_plot': selected_plot_json,
            'score_distribution_plot': distribution_plot_json
        }
        fig_serialized = json.dumps(response, cls=plotly.utils.PlotlyJSONEncoder)
        return Response(fig_serialized, mimetype='application/json')
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/node_info', methods=['POST'])
def node_info():
    data = request.get_json()
    filename = data.get('filename')
    node_name = data.get('node_name')
    if not filename or not node_name:
        return jsonify({'error': 'Missing filename or node name.'}), 400

    filepath = os.path.join(JSON_FOLDER, filename)
    if not os.path.exists(filepath):
        return jsonify({'error': 'File does not exist.'}), 400

    tree = load_tree_json(filepath)
    node_json = node_to_dict(tree, node_name)
    return jsonify({'node_info': node_json})

if __name__ == '__main__':
    app.run(debug=True)
