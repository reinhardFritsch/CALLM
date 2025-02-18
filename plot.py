import csv
import json
import os
from statistics import mean
from typing import Any, Dict, List, Optional, Tuple
from typing import List, Tuple, Dict, Optional
from anytree import Node, PreOrderIter
from plotly.subplots import make_subplots
import anytree
import igraph as ig
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from anytree import Node, PreOrderIter, find
from anytree.importer import JsonImporter
from tqdm import tqdm
import math
# Utility Functions
def insert_line_breaks(text: str, max_length: int = 100) -> str:
    """
    Inserts line breaks into the text to ensure it doesn't exceed the maximum length per line.

    Parameters:
    - text (str): The input text.
    - max_length (int): The maximum length of each line.

    Returns:
    - str: The text with inserted line breaks.
    """
    words = text.split(" ")
    current_line = ""
    lines = []

    for word in words:
        if len(current_line) + len(word) + 1 > max_length:
            lines.append(current_line)
            current_line = word
        else:
            current_line = f"{current_line} {word}".strip()
    if current_line:
        lines.append(current_line)
    return "<br>".join(lines)

# Data Classes
class NodeData:
    def __init__(
        self,
        category_name: str,
        sentences_new: List[str],
        sentences_old: List[str],
        similarity_score: float,
        matching_sentences: Optional[List[str]] = None,
        llm_comparator: Optional[Dict[str, Any]] = None,
        certainty_new: Optional[float] = None,
        certainty_old: Optional[float] = None
    ):
        self.category_name = category_name
        self.sentences_new = sentences_new
        self.sentences_old = sentences_old
        self.similarity_score = similarity_score
        self.matching_sentences = matching_sentences or []
        self.llm_comparator = llm_comparator or {}
        self.certainty_new = certainty_new
        self.certainty_old = certainty_old

    def __repr__(self):
        return (
            f"NodeData(category_name={self.category_name}, "
            f"sentences_new={self.sentences_new}, "
            f"certainty_new={self.certainty_new}, "
            f"sentences_old={self.sentences_old}, "
            f"certainty_old={self.certainty_old}, "
            f"similarity_score={self.similarity_score}, "
            f"matching_sentences={self.matching_sentences}, "
            f"llm_comparator={self.llm_comparator})"
        )

# Custom Importer
class CustomJsonImporter(JsonImporter):
    def __init__(self):
        super().__init__()

    def read(self, file):
        def dict_to_node(data: Dict[str, Any], parent: Optional[Node] = None) -> Node:
            node_data = None
            if data.get("data") is not None:
                node_data = NodeData(
                    category_name=data["data"].get("category_name", ""),
                    sentences_new=data["data"].get("sentences_new", []),
                    sentences_old=data["data"].get("sentences_old", []),
                    similarity_score=data["data"].get("similarity_score", 0.0),
                    matching_sentences=data["data"].get("matching_sentences", []),
                    llm_comparator=data["data"].get("llm_comparator", {}),
                    certainty_new=data["data"].get("certainty_new"),
                    certainty_old=data["data"].get("certainty_old")
                )
            node = Node(data.get("name", ""), parent=parent, data=node_data)
            for child_data in data.get("children", []):
                dict_to_node(child_data, parent=node)
            return node

        return dict_to_node(json.load(file))

# Load Tree from JSON
def load_tree_json(filename: str) -> Node:
    """
    Loads the tree structure from a JSON file.

    Parameters:
    - filename (str): The filename to load the tree from.

    Returns:
    - Node: The root node of the tree.
    """
    importer = CustomJsonImporter()
    with open(filename, "r") as f:
        tree = importer.read(f)
    return tree

# Node to Dictionary Conversion
def node_to_dict(tree: Node, node_name: str) -> str:
    """
    Converts a node's information to a dictionary.

    Parameters:
    - tree (Node): The root node of the tree.
    - node_name (str): The name of the node.

    Returns:
    - str: JSON-formatted string of the node's data.
    """
    target_node = find(tree, lambda n: n.name == node_name)
    if target_node and target_node.data:
        node_dict = {
            "category_name": target_node.name,
            "data": {
                "similarity_score": target_node.data.similarity_score,
                "certainty_new": target_node.data.certainty_new,
                "certainty_old": target_node.data.certainty_old,
                "llm_comparator": target_node.data.llm_comparator
            }
        }
        return json.dumps(node_dict, indent=4)
    else:
        return json.dumps({"error": "Node not found."}, indent=4)

# Save Plot
def save_plot(filename: str, plot_type: str, fig: go.Figure):
    """
    Saves the Plotly figure to an HTML file.

    Parameters:
    - filename (str): The original JSON filename.
    - plot_type (str): The type of plot.
    - fig (go.Figure): The Plotly figure to save.
    """
    output_plot_name = os.path.splitext(os.path.basename(filename))[0]
    output_folder = os.path.join("..", "plots", output_plot_name)
    os.makedirs(output_folder, exist_ok=True)
    fig.write_html(os.path.join(output_folder, f"{plot_type}_{output_plot_name}.html"))

# Plot Creation Functions
def create_tree_plot(filename: str) -> Dict[str, Any]:
    """
    Creates a Plotly Tree plot from a JSON tree file.

    Parameters:
    - filename (str): Path to the JSON file.

    Returns:
    - dict: Plotly figure as a JSON-serializable dictionary.
    """
    tree = load_tree_json(filename)
    G, nodes = tree_to_igraph(tree)

    layout = G.layout('rt')  # Radial tree layout
    position = {k: layout[k] for k in range(len(layout))}
    Y = [layout[k][1] for k in range(len(layout))]
    M = max(Y)

    E = [e.tuple for e in G.es]
    Xn = [position[k][0] for k in range(len(position))]
    Yn = [2 * M - position[k][1] for k in range(len(position))]

    Xe = []
    Ye = []
    for edge in E:
        Xe += [position[edge[0]][0], position[edge[1]][0], None]
        Ye += [2 * M - position[edge[0]][1], 2 * M - position[edge[1]][1], None]

    similarity_scores = [v['label'] for v in G.vs]
    similarity_scores[0] = calc_total_avg_similarity_score(tree)
    # TODO decide 
    similarity_scores[0] = 1.0
    max_score = 1.0#max(similarity_scores)
    inverted_scores = [max_score - value for value in similarity_scores]
    hover_texts = [
        f"Node: {G.vs[i]['name']}<br>Dissimilarity score: {inverted_scores[i]:.2f}"
        for i in range(len(G.vs))
    ]

    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=Xe, y=Ye,
        mode='lines',
        line=dict(color='rgb(0,0,0)', width=1),
        hoverinfo='none',
        opacity=1
    ))
    diverging_color_scale = [
        (0.0, "#0571b0"),  # Blue for low scores
        (0.5, "#f7f7f7"),  # White at the midpoint
        (1.0, "#ca0020"),  # Red for high scores
    ]

    fig.add_trace(go.Scatter(
        x=Xn, y=Yn,
        mode='markers',
        name='Nodes',
        marker=dict(
            symbol='circle',
            size=10,
            color=inverted_scores,
            colorscale=diverging_color_scale,
            coloraxis='coloraxis',
            line=dict(color='rgb(0,0,0)', width=1)
        ),
        text=hover_texts,
        hoverinfo='text',
        opacity=1,
        customdata=[v['name'] for v in G.vs]
    ))
    fig.update_layout(
        title=f'Tree for {os.path.splitext(os.path.basename(filename))[0]}',
        showlegend=False,
        xaxis=dict(showgrid=False, zeroline=False, visible=False),
        yaxis=dict(showgrid=False, zeroline=False, visible=False),
        coloraxis=dict(
            colorbar=dict(title='Dissimilarity score'),
            colorscale=diverging_color_scale
        ),
        height=800,
    )
    save_plot(filename, "tree", fig)
    return fig.to_plotly_json()

def create_score_distribution_plot(filename: str) -> Dict[str, Any]:
    """
    Creates a distribution plot (box plot) for the Dissimilarity scores.

    Parameters:
    - filename (str): Path to the JSON file.

    Returns:
    - dict: Plotly figure as a JSON-serializable dictionary.
    """
    tree = load_tree_json(filename)
    similarity_scores_weighted = [
        calculate_weighted_score_for_node(node) for node in PreOrderIter(tree)
        if hasattr(node.data, 'similarity_score') and node.data.similarity_score is not None
    ][1:]
    similarity_scores = [node.data.similarity_score for node in PreOrderIter(tree) if hasattr(node.data, 'similarity_score')][1:]
    fig = px.box(
        x=similarity_scores,
        labels={'x': 'Similarity Score'},
        title=f'Similarity Score Distribution {os.path.splitext(os.path.basename(filename))[0]}\nAvg weighted score: {calculate_weighted_average_score(tree):.5f}\n Avg similarity score: {mean(similarity_scores):.5f}',
        color_discrete_sequence=['#636EFA'],
    )
    fig.update_layout(
        xaxis_title="Similarity Score",
        yaxis_title="",
        title_font=dict(size=20),
        font=dict(size=14),
        margin=dict(t=50, l=25, r=25, b=25),
        plot_bgcolor='rgba(240,240,240,0.5)',
        height=200
    )
    fig.update_xaxes(
        gridcolor='rgba(200,200,200,0.5)',
        zeroline=True,
        zerolinewidth=2,
        zerolinecolor='rgba(150,150,150,0.8)',
    )
    save_plot(filename, "distribution", fig)
    return fig.to_plotly_json()

def create_treemap_plot(filename: str) -> Dict[str, Any]:
    """
    Creates a Plotly Treemap from a JSON tree file with a custom diverging color scale.

    Parameters:
    - filename (str): Path to the JSON file.

    Returns:
    - dict: Plotly figure as a JSON-serializable dictionary.
    """
    tree = load_tree_json(filename)

    def get_tree_paths(
        node: Node,
        node_name_list: Optional[List[str]] = None,
        parent_list: Optional[List[str]] = None,
        similarity_score_list: Optional[List[float]] = None,
        new: Optional[List[str]] = None,
        old: Optional[List[str]] = None
    ) -> Tuple[List[str], List[str], List[float], List[str], List[str]]:
        if node_name_list is None:
            node_name_list = []
        if parent_list is None:
            parent_list = []
        if similarity_score_list is None:
            similarity_score_list = []
        if new is None:
            new = []
        if old is None:
            old = []

        node_name_list.append(node.name)
        parent_list.append(node.parent.name if node.parent else "")
        similarity_score_list.append(node.data.similarity_score)
        new.append("<br>".join([insert_line_breaks(sentence) for sentence in node.data.sentences_new]))
        old.append("<br>".join([insert_line_breaks(sentence) for sentence in node.data.sentences_old]))

        for child in node.children:
            get_tree_paths(child, node_name_list, parent_list, similarity_score_list, new, old)

        return node_name_list, parent_list, similarity_score_list, new, old

    node_names, parents, similarity_scores, new_sentences, old_sentences = get_tree_paths(tree)
    similarity_scores[0] = calc_total_avg_similarity_score(tree)
    max_score = 1.0#max(similarity_scores)
    inverted_scores = [max(max_score - value, 0.01) for value in similarity_scores]

    # Define the custom diverging color scale
    diverging_color_scale = [
        (0.0, "#0571b0"),  # Blue at low end
        (0.5, "#f7f7f7"),  # White at midpoint
        (1.0, "#ca0020"),  # Red at high end
    ]

    fig = px.treemap(
        title=f'Treemap for {os.path.splitext(os.path.basename(filename))[0]}',
        names=node_names,
        parents=parents,
        values=inverted_scores,
        color=inverted_scores,
        color_continuous_scale=diverging_color_scale,  # Use the custom color scale
    )
    fig.update_traces(
        textfont_size=20,
        hovertemplate=(
            "<b>%{label}</b><br>"
            "<b>Score:</b> %{value}<br>"
        )
    )
    fig.update_layout(
        margin=dict(t=50, l=25, r=25, b=25),
        coloraxis_colorbar=dict(title='Dissimilarity score')
    )
    save_plot(filename, "treemap", fig)
    return fig.to_plotly_json()


def create_sunburst_plot(filename: str) -> Dict[str, Any]:
    """
    Creates a Plotly Sunburst Diagram from a JSON tree file.

    Parameters:
    - filename (str): Path to the JSON file.

    Returns:
    - dict: Plotly figure as a JSON-serializable dictionary.
    """
    tree = load_tree_json(filename)

    def get_tree_paths(
        node: Node,
        node_name_list: Optional[List[str]] = None,
        parent_list: Optional[List[str]] = None,
        similarity_score_list: Optional[List[float]] = None,
        new: Optional[List[str]] = None,
        old: Optional[List[str]] = None,
        rational_list: Optional[List[str]] = None,
        certainty_old: Optional[List[Optional[float]]] = None,
        certainty_new: Optional[List[Optional[float]]] = None
    ) -> Tuple[List[str], List[str], List[float], List[str], List[str], List[str], List[Optional[float]], List[Optional[float]]]:
        if node_name_list is None:
            node_name_list = []
        if parent_list is None:
            parent_list = []
        if similarity_score_list is None:
            similarity_score_list = []
        if new is None:
            new = []
        if old is None:
            old = []
        if rational_list is None:
            rational_list = []
        if certainty_old is None:
            certainty_old = []
        if certainty_new is None:
            certainty_new = []

        if node.parent is None:
            parent_list.append("")
            node_name_list.append(node.name)
            rational_list.append("")
            new.append("")
            old.append("")
            similarity_score_list.append(calc_total_avg_similarity_score(tree))
            certainty_new.append(None)
            certainty_old.append(None)
        else:
            node_name_list.append(node.name)
            parent_list.append(node.parent.name)
            certainty_old.append(node.data.certainty_old)
            certainty_new.append(node.data.certainty_new)
            rational_bullets = node.data.llm_comparator.get("rational_bullets", []) if node.data.llm_comparator else []
            numbered_bullets = [f"{i+1}. {insert_line_breaks(bullet)}" for i, bullet in enumerate(rational_bullets)]
            rational_list.append("<br>".join(numbered_bullets))
            new.append(insert_line_breaks("<b>output_new</b>: " + " ".join(node.data.sentences_new)))
            old.append(insert_line_breaks("<b>output_old</b>: " + " ".join(node.data.sentences_old)))
            similarity_score_list.append(node.data.similarity_score)

        for child in node.children:
            get_tree_paths(child, node_name_list, parent_list, similarity_score_list, new, old, rational_list, certainty_old, certainty_new)
        return node_name_list, parent_list, similarity_score_list, new, old, rational_list, certainty_old, certainty_new

    node_names, parents, similarity_scores, new_sents, old_sents, rational_list, certainty_old, certainty_new = get_tree_paths(tree)
    text_info = []
    mode = "reasoning"
    if mode == "name":
        avg_certainty = calc_total_avg_certainty_scores(tree)
        text_info.append(f"<b>{node_names[0]}<br>avg_similarity_score: {similarity_scores[0]:.2f}</b><br><b>avg_certainty_score: {avg_certainty:.2f}</b>")
        text_info.extend([
            f"<b>{nm}</b><br> <b>{sc:.2f}% similarity | certainty old/new: "
            f"{f'{co:.2f}' if co is not None else '(N/A)'}/"
            f"{f'{cn:.2f}' if cn is not None else '(N/A)'}</b>"
            for nm, sc, co, cn in zip(node_names[1:], similarity_scores[1:], certainty_old[1:], certainty_new[1:])
        ])
    elif mode == "all":
        text_info = [
            f"<b>{nm}</b>: <b>{sc:.2f}% similarity</b><br>{n}<br>{o}<br><b>Labels</b><br>{rl}"
            for n, o, nm, sc, rl in zip(new_sents, old_sents, node_names, similarity_scores, rational_list)
        ]
    elif mode == "reasoning":
        avg_certainty = calc_total_avg_certainty_scores(tree)
        text_info.append(f"<b>{node_names[0]}<br>avg_similarity_score: {similarity_scores[0]:.2f}</b><br><b>avg_certainty_score: {avg_certainty:.2f}</b>")
        text_info.extend([
            f"<b>{nm}</b><br> <b>{sc:.2f}% similarity | certainty old/new: "
            f"{f'{co:.2f}' if co is not None else '(N/A)'}/"
            f"{f'{cn:.2f}' if cn is not None else '(N/A)'}</b><br><br>{rl}"
            for nm, sc, rl, co, cn in zip(node_names[1:], similarity_scores[1:], rational_list[1:], certainty_old[1:], certainty_new[1:])
        ])

    similarity_scores[0] = calc_total_avg_similarity_score(tree)
    max_score = 1.0#max(similarity_scores)
    inverted_scores = [max(max_score - value, 0.01) for value in similarity_scores]
    hover_text = [f"{nm}<br>Dissimilarity score: {sc:.2f}" for nm, sc in zip(node_names, inverted_scores)]
    diverging_color_scale = [
        (0.0, "#0571b0"),  # Blue for low scores
        (0.5, "#f7f7f7"),  # White at the midpoint
        (1.0, "#ca0020"),  # Red for high scores
    ]
    fig = go.Figure(go.Sunburst(
        labels=node_names,
        parents=parents,
        values=inverted_scores,
        text=text_info,
        textinfo="text",
        marker=dict(colors=inverted_scores, coloraxis='coloraxis'),
        hoverinfo="text",
        hovertext=hover_text,
        customdata=node_names
    ))

    fig.update_layout(
        title=f'Sunburst for {os.path.splitext(os.path.basename(filename))[0]}',
        font=dict(family="Arial, sans-serif", size=16),
        paper_bgcolor="white",
        plot_bgcolor="white",
        width=2000,
        height=1200,
        margin=dict(t=50, l=25, r=25, b=25),
        coloraxis=dict(
            colorscale=diverging_color_scale,
            colorbar=dict(title='Dissimilarity score')
        )
    )
    fig.update_traces(marker=dict(line=dict(color='black', width=0.5)), textfont_size=100)
    save_plot(filename, "sunburst", fig)
    return fig.to_plotly_json()

def create_aggregation_plot(filename: str) -> Dict[str, Any]:
    """
    Creates a Plotly Tree Node Aggregation plot from a JSON tree file.

    Parameters:
    - filename (str): Path to the JSON file.

    Returns:
    - dict: Plotly figure as a JSON-serializable dictionary.
    """
    tree = load_tree_json(filename)

    def aggregate_scores(node: Node) -> float:
        if node.is_leaf:
            return node.data.similarity_score
        else:
            children_scores = sum(aggregate_scores(child) for child in node.children)
            avg_children_score = children_scores / len(node.children) if node.children else 0
            return node.data.similarity_score + avg_children_score

    def get_aggregated_scores(root: Node) -> Dict[str, float]:
        scores = {}
        for node in root.descendants + (root,):
            scores[node.name] = aggregate_scores(node)
        return scores

    aggregated_scores = get_aggregated_scores(tree)
    G, nodes = tree_to_igraph_aggregated(tree, aggregated_scores)

    layout = G.layout('rt')
    position = {k: layout[k] for k in range(len(layout))}
    Y = [layout[k][1] for k in range(len(layout))]
    M = max(Y)

    E = [e.tuple for e in G.es]
    Xn = [position[k][0] for k in range(len(position))]
    Yn = [2 * M - position[k][1] for k in range(len(position))]

    Xe = []
    Ye = []
    for edge in E:
        Xe += [position[edge[0]][0], position[edge[1]][0], None]
        Ye += [2 * M - position[edge[0]][1], 2 * M - position[edge[1]][1], None]

    aggregated_scores_array = np.array([v['label'] for v in G.vs])
    log_aggregated_scores = np.log1p(aggregated_scores_array)
    original_scores = [v['original_label'] for v in G.vs]
    hover_texts = [
        f"Node: {G.vs[i]['name']}<br>Original Score: {original_scores[i]:.2f}"
        for i in range(len(G.vs))
    ]

    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=Xe, y=Ye,
        mode='lines',
        line=dict(color='rgb(0,0,0)', width=1),
        hoverinfo='none',
        opacity=1
    ))
    fig.add_trace(go.Scatter(
        x=Xn, y=Yn,
        mode='markers',
        name='Nodes',
        marker=dict(
            symbol='circle-dot',
            size=10,
            color=log_aggregated_scores,
            colorscale='Reds',
            colorbar=dict(
                title='Aggregated Score',
                tickvals=[min(log_aggregated_scores), max(log_aggregated_scores)],
                ticktext=[f"{min(log_aggregated_scores):.2f}", f"{max(log_aggregated_scores):.2f}"]
            ),
            line=dict(color='rgb(50,50,50)', width=0.2)
        ),
        text=hover_texts,
        hoverinfo='text',
        opacity=1,
        customdata=[v['name'] for v in G.vs],
    ))
    fig.update_layout(
        showlegend=False,
        xaxis=dict(showgrid=False, zeroline=False, visible=False),
        yaxis=dict(showgrid=False, zeroline=False, visible=False),
        width=1900,
        height=1000
    )
    save_plot(filename, "aggregation", fig)
    return fig.to_plotly_json()

# Helper Functions for Tree and Graph Manipulations
def tree_to_igraph(root: Node) -> Tuple[ig.Graph, Dict[str, int]]:
    """
    Converts an AnyTree structure to an iGraph graph.

    Parameters:
    - root (Node): The root node of the AnyTree.

    Returns:
    - igraph.Graph: The resulting iGraph graph.
    - dict: A mapping from node names to graph vertex indices.
    """
    g = ig.Graph(directed=True)
    nodes = {}

    def add_edges(node: Node):
        if node.name not in nodes:
            nodes[node.name] = g.add_vertex(name=node.name, label=node.data.similarity_score)
        for child in node.children:
            if child.name not in nodes:
                nodes[child.name] = g.add_vertex(name=child.name, label=child.data.similarity_score)
            g.add_edge(nodes[node.name], nodes[child.name])
            add_edges(child)

    add_edges(root)
    return g, nodes

def tree_to_igraph_aggregated(root: Node, scores: Dict[str, float]) -> Tuple[ig.Graph, Dict[str, int]]:
    """
    Converts an AnyTree structure to an iGraph graph with aggregated scores.

    Parameters:
    - root (Node): The root node of the AnyTree.
    - scores (dict): A dictionary mapping node names to aggregated scores.

    Returns:
    - igraph.Graph: The resulting iGraph graph.
    - dict: A mapping from node names to graph vertex indices.
    """
    g = ig.Graph(directed=True)
    nodes = {}

    def add_edges(node: Node):
        if node.name not in nodes:
            g_node = g.add_vertex(
                name=node.name,
                label=scores[node.name],
                original_label=node.data.similarity_score
            )
            nodes[node.name] = g_node.index
        for child in node.children:
            if child.name not in nodes:
                g_child = g.add_vertex(
                    name=child.name,
                    label=scores[child.name],
                    original_label=child.data.similarity_score
                )
                nodes[child.name] = g_child.index
            g.add_edge(nodes[node.name], nodes[child.name])
            add_edges(child)

    add_edges(root)
    return g, nodes

def calc_total_avg_similarity_score(tree: Node) -> float:
    """
    Calculates the total average similarity score for the entire tree.

    Parameters:
    - tree (Node): The root node of the tree.

    Returns:
    - float: The total average similarity score.
    """
    similarity_scores = [
        node.data.similarity_score for node in PreOrderIter(tree) if hasattr(node.data, 'similarity_score')
    ]
    if not similarity_scores:
        return 0.0
    total_avg_score = sum(similarity_scores) / len(similarity_scores)
    return total_avg_score

def calc_total_avg_certainty_scores(tree: Node) -> float:
    """
    Calculates the total average certainty score (both old and new) for the entire tree.

    Parameters:
    - tree (Node): The root node of the tree.

    Returns:
    - float: The total average certainty score.
    """
    certainty_old_scores = [
        node.data.certainty_old for node in PreOrderIter(tree)
        if hasattr(node.data, 'certainty_old') and node.data.certainty_old is not None
    ]
    certainty_new_scores = [
        node.data.certainty_new for node in PreOrderIter(tree)
        if hasattr(node.data, 'certainty_new') and node.data.certainty_new is not None
    ]
    avg_certainty_old = sum(certainty_old_scores) / len(certainty_old_scores) if certainty_old_scores else 0.0
    avg_certainty_new = sum(certainty_new_scores) / len(certainty_new_scores) if certainty_new_scores else 0.0
    return (avg_certainty_old + avg_certainty_new) / 2.0



def get_all_paths_skip_root(root: Node, count: Optional[int] = None) -> List[Tuple[Node, ...]]:
    """
    Collect all possible paths in the tree, starting from depth 1 (immediate children of root),
    excluding the root node itself. Each path starts from a child of the root and extends to a leaf node.
    Additionally, each leaf node is included as a single-node path.

    Parameters:
    - root (Node): The root node of the tree.
    - count (Optional[int]): The maximum number of paths to return.

    Returns:
    - List[Tuple[Node, ...]]: A list of paths, where each path is a tuple of nodes starting from depth 1.
    """
    paths = []
    for child in root.children:
        if child.is_leaf:
            paths.append((child,))
            if count and len(paths) >= count:
                return paths
        else:
            for descendant in PreOrderIter(child):
                if descendant.is_leaf:
                    path = descendant.path[1:]  # Exclude root
                    paths.append(tuple(path))
                    if count and len(paths) >= count:
                        return paths
    return paths

def calculate_weighted_score_for_node(node: Node) -> float:
    """
    Calculate the weighted score for a given node based on its depth.

    Parameters:
    - node (Node): The node for which to calculate the weighted score.

    Returns:
    - float: The weighted score of the node.
    """
    decay_factor = 0.3
    similarity_score = 1 - getattr(getattr(node, 'data', None), 'similarity_score', 0.0)
    weight = math.exp(- decay_factor * node.depth)
    return similarity_score * weight

def calculate_weighted_average_score(node: Node) -> float:
    """
    Calculate the weighted average score based on the scores and their depths.

    Parameters:
    - scores (List[float]): The list of scores to calculate the weighted average for.

    Returns:
    - float: The weighted average score.
    """

    total_weighted_score = 0.0
    total_weights = 0.0
    decay_factor = 0.3
    for node in PreOrderIter(node):
        if node.is_root:
            continue
        weight = math.exp(- decay_factor * node.depth)
        similarity_score = getattr(getattr(node, 'data', None), 'similarity_score', 0.0)
        dissimilarity_score = 1.0 - similarity_score
        total_weighted_score += dissimilarity_score * weight
        total_weights += weight
    average_score = (total_weighted_score / total_weights) if total_weights > 0 else 0.0

    return average_score

def calculate_weighted_score_skip_root(path: Tuple[Node, ...]) -> float:
    """
    Calculate the weighted score for a given path, excluding the root node.

    Parameters:
    - path (Tuple[Node, ...]): A tuple of nodes representing the path starting from depth 1.
    - weights (Dict[int, float]): A dictionary mapping node depths to their weights.

    Returns:
    - float: The average weighted score of the path.
    """
    total_weighted_score = 0.0
    total_weights = 0.0
    decay_factor = 0.3
    for node in path:
        weight = math.exp(- decay_factor * node.depth)
        similarity_score = getattr(getattr(node, 'data', None), 'similarity_score', 0.0)
        dissimilarity_score = 1.0 - similarity_score
        total_weighted_score += dissimilarity_score * weight
        total_weights += weight
    average_score = (total_weighted_score / total_weights) if total_weights > 0 else 0.0
    return average_score

def get_paths_sorted_by_lowest_score_skip_root(root: Node) -> List[Tuple[List[str], float]]:
    """
    Get all paths (excluding root) sorted by their weighted scores from lowest to highest.

    Parameters:
    - root (Node): The root node of the tree.

    Returns:
    - List[Tuple[List[str], float]]: A list of tuples containing path names and their weighted scores, sorted by score.
    """
    paths = get_all_paths_skip_root(root)
    path_scores = []
    for path in paths:
        score = calculate_weighted_score_skip_root(path)
        path_names = [node.name for node in path]
        path_scores.append((path_names, score))
    sorted_paths = sorted(path_scores, key=lambda x: x[1], reverse=True)
    return sorted_paths

def find_node_depth(root: Node, name: str) -> Optional[int]:
    """
    Find the depth of a node with the given name.

    Parameters:
    - root (Node): The root node of the tree.
    - name (str): The name of the node to find.

    Returns:
    - Optional[int]: The depth of the node if found, else None.
    """
    for node in PreOrderIter(root):
        if node.name == name:
            return node.depth
    return None

def build_anytree_directly(paths_with_scores: List[Tuple[List[str], float]]) -> List[Node]:
    """
    Builds anytree Nodes directly from a list of paths with similarity scores.

    Parameters:
    - paths_with_scores (List[Tuple[List[str], float]]): List of tuples with path and score.

    Returns:
    - List[Node]: List of root nodes.
    """
    top_level_nodes = {}
    for path, score in paths_with_scores:
        if not path:
            continue
        first_node_name = path[0]
        if first_node_name in top_level_nodes:
            node = top_level_nodes[first_node_name]
            node.count += 1
        else:
            node = Node(name=first_node_name, count=1, score=score)
            top_level_nodes[first_node_name] = node
        current_node = node
        for part in path[1:]:
            existing_children = {child.name: child for child in current_node.children}
            if part in existing_children:
                child_node = existing_children[part]
                child_node.count += 1
            else:
                child_node = Node(name=part, count=1, parent=current_node, score=score)
            current_node = child_node
    return list(top_level_nodes.values())

def get_similarity_score(root: Node, target_name: str) -> Optional[float]:
    """
    Get the similarity score of a node and adjust it based on depth.

    Parameters:
    - root (Node): The root node of the tree.
    - target_name (str): The name of the target node.

    Returns:
    - Optional[float]: Adjusted similarity score if node is found, else None.
    """
    decay_factor = 0.9
    for node in PreOrderIter(root):
        if node.name == target_name:
            base_score = getattr(getattr(node, 'data', None), 'similarity_score', 0.0)
            adjusted_score = (1 - base_score) * (decay_factor ** node.depth)
            return adjusted_score
    return None

def anytree_to_igraph(anytree_root: Node) -> ig.Graph:
    """
    Converts an AnyTree tree to an igraph Graph.
    Edge thickness is based on child node's count.

    Parameters:
    - anytree_root (Node): The root node of the AnyTree.

    Returns:
    - ig.Graph: igraph Graph object with nodes and edges from the AnyTree.
    """
    g = ig.Graph(directed=True)
    node_to_index = {}
    for node in PreOrderIter(anytree_root):
        g.add_vertex(name=node.name, count=node.count, score=node.score)
        node_to_index[node] = g.vcount() - 1
        if not node.is_root:
            parent_index = node_to_index[node.parent]
            child_index = node_to_index[node]
            g.add_edge(parent_index, child_index, thickness=node.count)
    return g

def plot_igraph_with_plotly(
    g: ig.Graph, 
    title: str = "Tree Visualization", 
    global_min_score: Optional[float] = None, 
    global_max_score: Optional[float] = None
) -> Dict[str, Any]:
    """
    Plots an igraph Graph using Plotly with edge thickness based on edge attributes.
    All plots will use the shared color scale for the node scores.
    
    Parameters:
    - g: igraph.Graph object.
    - title: Title of the plot.
    - global_min_score: Global minimum score for color scale.
    - global_max_score: Global maximum score for color scale.
    
    Returns:
    - Dict[str, Any]: Plotly Figure object as a JSON-serializable dictionary.
    """
    layout = g.layout("kamada_kawai")
    x_coords, y_coords = zip(*layout.coords)
    index_to_position = {idx: (coord[0], coord[1]) for idx, coord in enumerate(layout.coords)}

    fig = go.Figure()


    # Add edges
    for edge in g.es:
        source, target = edge.source, edge.target
        thickness = edge['thickness']

        x0, y0 = index_to_position[source]
        x1, y1 = index_to_position[target]

        fig.add_shape(
            type="line",
            x0=x0,
            y0=y0,
            x1=x1,
            y1=y1,
            line=dict(
                color="#000000",
                #width=thickness
                width=3
            )
        )
        fig.add_annotation(
            ax=x0,
            ay=y0,
            axref="x",
            ayref="y",
            x=x1,
            y=y1,
            xref="x",
            yref="y",
            showarrow=True,
            arrowhead=2,
            arrowsize=2,
            #arrowwidth=thickness,
            arrowcolor="#000000"
        )

    scores = [v['score'] for v in g.vs]
    score_size = scale_node_sizes(scores)

    node_x = x_coords
    node_y = y_coords
    node_text_hover = [
        f"Name: {v['name']}<br>Count: {v['count']}<br>Score: {v['score']}"
        for v in g.vs
    ]
    node_text = [v['name'] for v in g.vs]

    fig.add_trace(
        go.Scatter(
            x=node_x,
            y=node_y,
            mode='markers',
            text=node_text,
            hoverinfo='text',
            hovertext=node_text_hover,
            customdata=node_text,
            marker=dict(
                line = dict(width=2, color='black'),
                showscale=True,
                colorscale='Reds',
                color=scores,
                size=score_size,
                colorbar=dict(
                    thickness=15,
                    title='Node Score',
                    xanchor='left',
                    titleside='right',
                ),
                line_width=2,
                coloraxis='coloraxis',  # Use shared coloraxis
            )
        )
    )

    # Set coloraxis in layout with global min and max
    fig.update_layout(
        coloraxis=dict(
            colorscale='Reds',
            colorbar=dict(
                title='Node Score',
            ),
            cmin=global_min_score,
            cmax=global_max_score
        )
    )
    fig.update_layout(
        title=title,
        titlefont=dict(size=20, color='black'),
        showlegend=False,
        hovermode='closest',
        margin=dict(b=20, l=5, r=5, t=50),
        paper_bgcolor='white',
        plot_bgcolor='white',
        xaxis=dict(showgrid=False, zeroline=False, visible=True,showticklabels=False,showline=True, linewidth=2,linecolor='black',mirror=True ),
        yaxis=dict(showgrid=False, zeroline=False, visible=True,showticklabels=False,showline=True, linewidth=2,linecolor='black',mirror=True),
        #height=1000
    )
    save_plot("tree", "tree", fig)
    return fig.to_plotly_json()
MIN_NODE_SIZE = 10
MAX_NODE_SIZE = 30
def scale_node_sizes(scores, min_size=MIN_NODE_SIZE, max_size=MAX_NODE_SIZE):
    min_score = min(scores)
    max_score = max(scores)
    
    # Handle the case where all scores are the same
    if max_score == min_score:
        return [ (min_size + max_size) / 2 ] * len(scores)
    
    scaled_sizes = []
    for s in scores:
        # Linear scaling
        size = min_size + (s - min_score) / (max_score - min_score) * (max_size - min_size)
        scaled_sizes.append(size)
    return scaled_sizes


def fill_in_score(default_tree: Node, to_fill_tree: Node) -> Node:
    """
    Fill in the scores for the subtree nodes based on the default tree.

    Parameters:
    - default_tree (Node): The original full tree.
    - to_fill_tree (Node): The subtree to fill scores for.

    Returns:
    - Node: The subtree with filled scores.
    """
    for node in PreOrderIter(to_fill_tree):
        node.score = get_similarity_score(default_tree, node.name)
    return to_fill_tree

def create_subtree_plot(filename: str) -> Dict[str, Any]:
    """
    Creates a Subtree Plot from a JSON tree file, combining multiple subtrees into a single Plotly figure with subplots.
    All subplots share the same color scale for node scores.
    
    Parameters:
    - filename (str): Path to the JSON file.
    
    Returns:
    - Dict[str, Any]: Plotly figure as a JSON-serializable dictionary.
    """
    tree = load_tree_json(filename)
    sorted_paths = get_paths_sorted_by_lowest_score_skip_root(tree)
    top_elements = max(int(len(sorted_paths) * 0.1), 10)
    sorted_paths = sorted_paths[:top_elements]
    anytree_roots = build_anytree_directly(sorted_paths)
    figures = []
    all_scores = []

    # Collect all scores from all subtrees to determine global min and max
    for sub_tree in anytree_roots:
        filled_tree = fill_in_score(tree, sub_tree)
        scores = [node.score for node in PreOrderIter(filled_tree) if node.score is not None]
        all_scores.extend(scores)

    if not all_scores:
        return {}

    global_min_score = min(all_scores)
    global_max_score = max(all_scores)

    # Generate individual figures for each subtree
    for sub_tree in tqdm(anytree_roots, desc="Processing subtrees"):
        filled_tree = fill_in_score(tree, sub_tree)
        g = anytree_to_igraph(filled_tree)
        fig_json = plot_igraph_with_plotly(
            g, 
            title=filled_tree.name, 
            global_min_score=global_min_score, 
            global_max_score=global_max_score
        )
        figures.append(fig_json)

    num_plots = len(figures)
    cols = 1  # Arrange subplots in a single column
    rows = num_plots

    per_subplot_height = 500
    width = 1000
    total_height = per_subplot_height * rows
    # Create a combined figure with subplots
    combined_fig = make_subplots(
        rows=rows,
        cols=cols,
        subplot_titles=[fig['layout']['title']['text'] for fig in figures],
        vertical_spacing=0.05
    )

    for idx, fig_json in enumerate(figures):
        row = idx + 1
        col = 1
        x_axis = f'x{row}' if row > 1 else 'x'
        y_axis = f'y{row}' if row > 1 else 'y'

        # Add traces
        for trace in fig_json['data']:
            # Ensure all traces use the shared coloraxis
            if 'marker' in trace and 'coloraxis' in trace['marker']:
                trace['marker']['coloraxis'] = 'coloraxis'
            combined_fig.add_trace(go.Scatter(**trace), row=row, col=col)

        # Add shapes (edges)
        for shape in fig_json['layout'].get('shapes', []):
            # Update xref and yref to correspond to the subplot's axes
            updated_shape = shape.copy()
            updated_shape['xref'] = x_axis
            updated_shape['yref'] = y_axis
            updated_shape['layer'] = 'below'  # Ensure shapes are below axis lines
            combined_fig.add_shape(updated_shape, row=row, col=col)

        # Add annotations (arrows)
        for annotation in fig_json['layout'].get('annotations', []):
            # Update xref and yref to correspond to the subplot's axes
            updated_annotation = annotation.copy()
            # Annotations have multiple references: xref, yref, axref, ayref
            # They should all point to the subplot's axes
            updated_annotation['xref'] = x_axis
            updated_annotation['yref'] = y_axis
            updated_annotation['axref'] = x_axis
            updated_annotation['ayref'] = y_axis
            combined_fig.add_annotation(updated_annotation, row=row, col=col)

    # Update layout with shared coloraxis settings
    combined_fig.update_layout(
        coloraxis=dict(
            colorscale='Reds',
            colorbar=dict(
                title='Node Score',
            ),
            cmin=global_min_score,
            cmax=global_max_score
        ),
        title=f'Subtrees for {os.path.splitext(os.path.basename(filename))[0]}',
        showlegend=False,
        width=width,
        height=total_height,  # Adjust height based on the number of rows
        margin=dict(t=100, l=50, r=50, b=50),
        paper_bgcolor='white',
        plot_bgcolor="#D3D3D3",
    )

    # Update axes for each subplot to ensure axis lines are visible
    for row in range(1, rows + 1):
        x_axis = f'x{row}' if row > 1 else 'x'
        y_axis = f'y{row}' if row > 1 else 'y'
        combined_fig.update_xaxes(
            showgrid=True, 
            zeroline=False, 
            visible=True,
            showline=True, 
            linewidth=2, 
            linecolor='black',
            mirror=True,
            row=row,
            col=col,
            showticklabels=False
        )
        combined_fig.update_yaxes(
            showgrid=True, 
            zeroline=False, 
            visible=True,
            showline=True, 
            linewidth=2, 
            linecolor='black',
            mirror=True,
            row=row,
            col=col,
            showticklabels=False
        )

    # Update annotations (subplot titles)
    for annotation in combined_fig.layout.annotations:
        annotation.font.color = 'black'
        annotation.font.size = 14

    save_plot(filename, "subtree", combined_fig)
    return combined_fig.to_plotly_json()


####################
# Spider diagram   #
####################

import google.generativeai as genai
import re
import xml.etree.ElementTree as ET
from collections import defaultdict
from typing import Dict, List, Union
import plotly.graph_objects as go
from anytree import Node, PreOrderIter
import os
from tqdm import tqdm

# Configuration
safety_settings = [
    {
        "category": "HARM_CATEGORY_DANGEROUS",
        "threshold": "BLOCK_NONE",
    },
    {
        "category": "HARM_CATEGORY_HARASSMENT",
        "threshold": "BLOCK_NONE",
    },
    {
        "category": "HARM_CATEGORY_HATE_SPEECH",
        "threshold": "BLOCK_NONE",
    },
    {
        "category": "HARM_CATEGORY_SEXUALLY_EXPLICIT",
        "threshold": "BLOCK_NONE",
    },
    {
        "category": "HARM_CATEGORY_DANGEROUS_CONTENT",
        "threshold": "BLOCK_NONE",
    }
]

# Initialize Gemini
GOOGLE_API_KEY = "Your API Key"
gemini_model = "gemini-1.5-flash-8b-001"
genai.configure(api_key=GOOGLE_API_KEY)
model_gemini = genai.GenerativeModel(model_name=gemini_model, safety_settings=safety_settings)

def generate_labels(root_topic: str) -> List[str]:
    """Generate labels for classification using Gemini."""
    label_generation = '''
        Generate a list of high-level labels to classify subcategories under the root category of the input.
        The labels should focus on broad, overarching themes or dimensions of the topic. 
        Avoid specific technical terms or subfields, and instead, use general themes.
        The output should be in xml format, containing only the labels wrapped in <labels><Label>first label</Label></labels> tags. 
        Provide 10-15 labels.
    '''

    messages_label = [
        {"role": "model", "parts": label_generation},
        {"role": "user", "parts": root_topic}
    ]

    response = model_gemini.generate_content(
        messages_label,
        generation_config=genai.types.GenerationConfig(
            candidate_count=1,
            temperature=0.1
        )
    )
    
    # Extract XML from response
    match = re.search(r'```xml\n(.*?)\n```', response.text, re.DOTALL)
    if match:
        cleaned_text = match.group(1).replace("\\n", "\n").replace("&", "&amp;")
        root = ET.fromstring(cleaned_text)
        labels = [label.text for label in root.findall("Label")]
        return labels
    else:
        return ["General", "Other"]
import time
def classify_text_gemini(text: str, labels: List[str]) -> List[str]:
    """Classify text into one or more labels using Gemini."""
    prompt = f"Classify the following input text into one of the following categories:\n{', '.join(labels)}\nIt is also possible that the input text belongs to more than one label. Just return the label/s in a list."
    
    messages_label = [
        {"role": "model", "parts": [prompt]},
        {"role": "user", "parts": [text]}
    ]
    time.sleep(2)
    response = model_gemini.generate_content(
        messages_label,
        generation_config=genai.types.GenerationConfig(
            candidate_count=1,
            top_p=0.01
        )
    )
    
    response_text = response.text.strip()
    matched_categories = [label for label in labels if re.search(rf'\b{re.escape(label)}\b', response_text)]
    return matched_categories if matched_categories else ["Other"]

def aggregate_classifications(root: Node, model_type: str, labels: List[str]) -> Dict[str, int]:
    """Aggregate classifications for all nodes in the tree."""
    label_counts = defaultdict(int)
    nodes = [node for node in PreOrderIter(root) if not node.is_root]

    for node in tqdm(nodes, desc=f"Processing nodes for {model_type}"):
        if model_type == "new":
            if not node.data.certainty_new and node.data.sentences_new:
                text = node.data.sentences_new[0]
                category_name = node.name
                query = f"{category_name}: {text}"
                categories = classify_text_gemini(query, labels)
                for category in categories:
                    label_counts[category] += 1
        elif model_type == "old":
            if not node.data.certainty_old and node.data.sentences_old:
                text = node.data.sentences_old[0]
                category_name = node.name
                query = f"{category_name}: {text}"
                categories = classify_text_gemini(query, labels)
                for category in categories:
                    label_counts[category] += 1

    return label_counts

def get_labels_parallel(filename: str):
    """Process the tree and generate labels serially."""
    if not os.path.exists(filename):
        print(f"JSON file not found at path: {filename}")
        return {}, {}, []

    root = load_tree_json(filename)
    labels = generate_labels(root.name)
    
    # Process new and old data sequentially
    label_counts_new = aggregate_classifications(root, "new", labels)
    label_counts_old = aggregate_classifications(root, "old", labels)

    return label_counts_new, label_counts_old, labels

def create_spider_chart(
    label_counts_new: Dict[str, float],
    label_counts_old: Dict[str, float],
    labels: List[str],
    visible_traces: Dict[str, bool]
) -> go.Figure:
    """Create a spider/radar chart comparing two datasets and their differences."""
    # Create lists of values for each label
    values_new = [label_counts_new.get(label, 0) for label in labels]
    values_old = [label_counts_old.get(label, 0) for label in labels]
    
    # Calculate totals
    total_new = sum(values_new)
    total_old = sum(values_old)
    
    # Convert to percentages
    values_new_pct = [(v / total_new * 100) if total_new > 0 else 0 for v in values_new]
    values_old_pct = [(v / total_old * 100) if total_old > 0 else 0 for v in values_old]
    
    # Calculate percentage point differences
    differences = [new - old for new, old in zip(values_new_pct, values_old_pct)]
    values_diff = [abs(d) for d in differences]
    
    # Create difference direction labels
    diff_text = [
        f"New: {new:.1f}% vs Old: {old:.1f}%\n({'↑' if d > 0 else '↓' if d < 0 else '='}{abs(d):.1f}pp)"
        for new, old, d in zip(values_new_pct, values_old_pct, differences)
    ]
    
    # Close the polygons
    labels_closed = labels + [labels[0]]
    values_new_closed = values_new_pct + [values_new_pct[0]]
    values_old_closed = values_old_pct + [values_old_pct[0]]
    values_diff_closed = values_diff + [values_diff[0]]
    diff_text_closed = diff_text + [diff_text[0]]
    values_new_orig_closed = values_new + [values_new[0]]
    values_old_orig_closed = values_old + [values_old[0]]
    
    # Calculate max value for axis scaling
    max_value = max(
        max(values_new_closed if visible_traces.get('new', True) else [0]),
        max(values_old_closed if visible_traces.get('old', True) else [0]),
        max(values_diff_closed if visible_traces.get('diff', True) else [0])
    )
    
    # Define hover templates
    hovertemplate_new = "Label: %{theta}<br>New: %{r:.1f}%<br>Count: %{customdata}<br>Total: " + f"{total_new:,}<extra></extra>"
    hovertemplate_old = "Label: %{theta}<br>Old: %{r:.1f}%<br>Count: %{customdata}<br>Total: " + f"{total_old:,}<extra></extra>"
    hovertemplate_diff = "Label: %{theta}<br>Difference: %{r:.1f}pp<br>%{text}<extra></extra>"
    
    # Create figure
    fig = go.Figure()
    
    # Add traces
    fig.add_trace(go.Scatterpolar(
        r=values_new_closed,
        theta=labels_closed,
        fill='toself',
        name=f'New Data (Total: {total_new:,})',
        line_color='blue',
        opacity=0.5,
        line_width=1,
        visible='legendonly' if not visible_traces.get('new', True) else True,
        customdata=values_new_orig_closed,
        hovertemplate=hovertemplate_new
    ))
    
    fig.add_trace(go.Scatterpolar(
        r=values_old_closed,
        theta=labels_closed,
        fill='toself',
        name=f'Old Data (Total: {total_old:,})',
        line_color='green',
        opacity=0.5,
        line_width=1,
        visible='legendonly' if not visible_traces.get('old', True) else True,
        customdata=values_old_orig_closed,
        hovertemplate=hovertemplate_old
    ))
    
    fig.add_trace(go.Scatterpolar(
        r=values_diff_closed,
        theta=labels_closed,
        fill='toself',
        name='Differences (percentage points)',
        line_color='red',
        opacity=0.5,
        line_width=1,
        visible='legendonly' if not visible_traces.get('diff', True) else True,
        customdata=values_diff_closed,
        text=diff_text_closed,
        textposition='top center',
        marker=dict(size=8),
        hovertemplate=hovertemplate_diff
    ))
    
    # Update layout
    fig.update_layout(
        polar=dict(
            radialaxis=dict(
                visible=True,
                range=[0, max_value * 1.05],
                tickvals=[],
                showticklabels=False,
            ),
            angularaxis=dict(
                tickfont=dict(size=20, color='black')
            )
        ),
        showlegend=True,
        title=f"Label Distribution Comparison<br><sub>New Total: {total_new:,} vs Old Total: {total_old:,}</sub>",
        width=1800,
        height=1000
    )
    
    return fig