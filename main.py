from transformers import AutoModelForCausalLM, AutoTokenizer
import torch
from huggingface_hub import login
import gradio as gr
import io
import plotly.graph_objects as go
import networkx as nx
from PIL import Image
from dotenv import load_dotenv  
import os

load_dotenv(override=True)

login(token=os.getenv("HUGGINGFACE_TOKEN"))

models_info = {
    "google/gemma-3-1b-pt": "▁",
    "Qwen/Qwen2.5-1.5B": "Ġ",
}

current_model_name = list(models_info.keys())[0]
current_model_prefix = models_info[current_model_name]

model = AutoModelForCausalLM.from_pretrained(current_model_name)
tokenizer = AutoTokenizer.from_pretrained(current_model_name)

def load_model(model_name):
    global model, tokenizer, current_model_prefix
    current_model_prefix = models_info[model_name]
    model = AutoModelForCausalLM.from_pretrained(model_name)
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    return f"Loaded model: {model_name}"

def get_next_word_probabilities(input_sentence, top_k=100):
    inputs = tokenizer(input_sentence, return_tensors="pt")
    outputs = model(**inputs)
    logits = outputs.logits[:, -1, :]
    top_k_logits, top_k_indices = torch.topk(logits, k=top_k)
    top_k_words = tokenizer.convert_ids_to_tokens(top_k_indices[0])
    probabilities = torch.nn.functional.softmax(top_k_logits[0], dim=0)
    return list(zip(top_k_words, probabilities.tolist()))

def get_next_word_probabilities(input_sentence, top_k=5):
    inputs = tokenizer(input_sentence, return_tensors="pt")
    outputs = model(**inputs)
    logits = outputs.logits[:, -1, :]
    top_k_logits, top_k_indices = torch.topk(logits, k=top_k)
    top_k_words = tokenizer.convert_ids_to_tokens(top_k_indices[0])
    probabilities = torch.nn.functional.softmax(top_k_logits[0], dim=0)
    cleaned_words = [word.replace(current_model_prefix, '') for word in top_k_words]
    return list(zip(cleaned_words, probabilities.tolist()))

def build_prediction_tree(input_text, depth=2, breadth=3, use_pruning=False, pruning_threshold=0.05):
    """Build a tree of word predictions with specified depth and breadth."""
    tree = {"text": input_text, "prob": 1.0, "children": []}
    
    def build_subtree(node, current_depth):
        if current_depth >= depth:
            return
        
        predictions = get_next_word_probabilities(node["text"], top_k=100)[:breadth]
        
        for word, prob in predictions:
            if use_pruning and prob < pruning_threshold:
                continue
                
            new_text = f"{node['text']} {word}"
            child = {"text": new_text, "word": word, "prob": prob, "children": []}
            node["children"].append(child)
            build_subtree(child, current_depth + 1)
    
    build_subtree(tree, 0)
    return tree

def visualize_tree_plotly(tree, significant_digits=4):
    """Visualize the prediction tree using Plotly for interactive visualization."""
    G = nx.DiGraph()
    
    def add_nodes_edges(node, parent_id=None):
        node_id = id(node)
        if parent_id is None:
            label = node["text"]
            G.add_node(node_id, label=label, level=0, is_root=True)
        else:
            label = f"{node['word']} ({node['prob']:.{significant_digits}f})"
            level = G.nodes[parent_id]['level'] + 1
            G.add_node(node_id, label=label, level=level, is_root=False)
            G.add_edge(parent_id, node_id, weight=node["prob"])
        
        for child in node.get("children", []):
            add_nodes_edges(child, node_id)
    
    add_nodes_edges(tree)
    level_counts = {}
    for node, attrs in G.nodes(data=True):
        level = attrs['level']
        if level not in level_counts:
            level_counts[level] = 0
        level_counts[level] += 1
    pos = {}
    max_level = max(level_counts.keys()) if level_counts else 0
    
    def assign_y_coords(node_id, y_offset=0, level_width=1.0):
        children = list(G.successors(node_id))
        if not children:
            pos[node_id] = (G.nodes[node_id]['level'], y_offset)
            return y_offset + level_width, level_width
        
        total_width = 0
        child_widths = []
        for child in children:
            child_count = sum(1 for _ in nx.descendants(G, child)) + 1
            width = max(1.0, child_count * 0.8)
            child_widths.append(width)
            total_width += width
        current_y = y_offset
        for i, child in enumerate(children):
            next_y, _ = assign_y_coords(child, current_y, child_widths[i])
            current_y = next_y

        if children:
            child_ys = [pos[child][1] for child in children]
            center_y = sum(child_ys) / len(child_ys)
            pos[node_id] = (G.nodes[node_id]['level'], center_y)
        else:
            pos[node_id] = (G.nodes[node_id]['level'], y_offset)
        
        return y_offset + total_width, total_width

    root = [n for n, d in G.nodes(data=True) if d['is_root']][0]
    assign_y_coords(root)

    for node in pos:
        x, y = pos[node]
        pos[node] = (x * 1.5, y)

    edge_traces = []
    for edge in G.edges():
        x0, y0 = pos[edge[0]]
        x1, y1 = pos[edge[1]]
        weight = G.edges[edge]['weight']
        
        mid_x = (x0 + x1) / 2
        mid_y = (y0 + y1) / 2
  
        edge_trace = go.Scatter(
            x=[x0, mid_x, x1, None],
            y=[y0, mid_y, y1, None],
            line=dict(
                width=max(1, weight*5),
                color=f'rgba(50,50,50,{min(0.8, max(0.2, weight))})'  
            ),
            hoverinfo='none',
            mode='lines'
        )
        edge_traces.append(edge_trace)

    node_trace_root = go.Scatter(
        x=[],
        y=[],
        text=[],
        mode='markers+text',
        hoverinfo='text',
        marker=dict(
            color='lightblue',
            size=40,
            line=dict(width=2, color='blue')
        ),
        textposition="bottom center"
    )
    
    node_trace = go.Scatter(
        x=[],
        y=[],
        text=[],
        mode='markers+text',
        hoverinfo='text',
        marker=dict(
            color=[],
            size=30,
            colorscale='Blues',
            line=dict(width=2, color='blue')
        ),
        textposition="top right"
    )

    colors = []
    for node in G.nodes():
        x, y = pos[node]
        if G.nodes[node]['is_root']:
            node_trace_root.x = node_trace_root.x + (x,)
            node_trace_root.y = node_trace_root.y + (y,)
            node_trace_root.text = node_trace_root.text + (G.nodes[node]['label'],)
        else:
            node_trace.x = node_trace.x + (x,)
            node_trace.y = node_trace.y + (y,)
            node_trace.text = node_trace.text + (G.nodes[node]['label'],)
            parent = list(G.predecessors(node))[0]
            prob = G.edges[(parent, node)]['weight']
            colors.append(prob)
    
    node_trace.marker.color = colors
    

    max_nodes_at_level = max(level_counts.values()) if level_counts else 1
    base_width = 800
    base_height = 600
    width = base_width + (max_level * 150)  
    height = base_height + (max_nodes_at_level * 40)  
    width = max(width, 1200)
    height = max(height, 800)
    
    fig = go.Figure(data=edge_traces + [node_trace, node_trace_root],
                    layout=go.Layout(
                        showlegend=False,
                        hovermode='closest',
                        margin=dict(b=20, l=5, r=5, t=40),
                        xaxis=dict(
                            showgrid=False, 
                            zeroline=False, 
                            showticklabels=False,
                            range=[-0.5, max_level * 1.5 + 0.5] 
                        ),
                        yaxis=dict(
                            showgrid=False, 
                            zeroline=False, 
                            showticklabels=False,
                            scaleanchor="x",  
                            scaleratio=1 
                        ),
                        plot_bgcolor='rgba(255,255,255,1)',
                        title=f"Word Prediction Tree for: '{tree['text']}'",
                        width=width, 
                        height=height,
                        autosize=False
                    ))
    
    img_bytes = fig.to_image(format="png", width=width, height=height, scale=2)
    buf = io.BytesIO(img_bytes)
    img = Image.open(buf)
    
    return img

def predict_word_tree(model_choice, input_text, tree_depth, tree_breadth, use_pruning, pruning_threshold, significant_digits=4):
    global current_model_name
    if model_choice != current_model_name:
        current_model_name = model_choice
        load_model(model_choice)
    
    if not input_text.strip():
        return "Please enter some text", None
    
    try:
        tree = build_prediction_tree(input_text, depth=tree_depth, breadth=tree_breadth, 
                                    use_pruning=use_pruning, pruning_threshold=pruning_threshold)
        tree_image = visualize_tree_plotly(tree, significant_digits=significant_digits)
        summary = f"Word prediction tree for: '{input_text}'\n"
        summary += f"Depth: {tree_depth}, Breadth: {tree_breadth}\n"
        if use_pruning:
            summary += f"Pruning enabled with threshold: {pruning_threshold}\n"
        summary += "\nTop predictions:\n"
        for child in tree["children"]:
            summary += f"• {child['word']}: {child['prob']:.{significant_digits}f}\n"
        
        return summary, tree_image
    except Exception as e:
        return f"Error generating tree: {str(e)}", None


with gr.Blocks() as demo:
    gr.Markdown("# Word Prediction Tree")
    gr.Markdown("Enter text and explore possible word completions as a tree. (topk=100)")
    
    with gr.Row():
        with gr.Column(scale=1):
            model_dropdown = gr.Dropdown(
                choices=list(models_info.keys()),
                value=current_model_name,
                label="Select Model"
            )
            
            input_text = gr.Textbox(
                placeholder="Enter text here...",
                label="Input Text",
                value="Just do",
                lines=5
            )
            
            with gr.Row():
                tree_depth = gr.Slider(
                    minimum=1,
                    maximum=100,
                    value=6,
                    step=1,
                    label="Tree Depth"
                )
                
                tree_breadth = gr.Slider(
                    minimum=1,
                    maximum=4,
                    value=2,
                    step=1,
                    label="Branching Factor (Breadth)"
                )
   
            with gr.Row():
                use_pruning = gr.Checkbox(
                    label="Enable Probability Pruning",
                    value=False
                )
                
                pruning_threshold = gr.Slider(
                    minimum=0.01,
                    maximum=0.5,
                    value=0.05,
                    step=0.01,
                    label="Pruning Threshold"
                )         
            
            with gr.Row():
                significant_digits = gr.Slider(
                    minimum=1,
                    maximum=10,
                    value=4,
                    step=1,
                    label="Significant Digits"
                )
            
            predict_btn = gr.Button("Generate Tree")
            
            summary_output = gr.Textbox(label="Summary")
        
        with gr.Column(scale=2):
            image_output = gr.Image(label="Word Prediction Tree", type="pil")
    
    predict_btn.click(
        fn=predict_word_tree,
        inputs=[model_dropdown, input_text, tree_depth, tree_breadth, use_pruning, pruning_threshold, significant_digits],
        outputs=[summary_output, image_output]
    )

if __name__ == "__main__":
    demo.launch(server_name="0.0.0.0")