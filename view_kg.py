import networkx as nx
import matplotlib.pyplot as plt

# 1) Point to your GML file:
gml_filename = "knowledge_graph.gml"

# 2) Read the graph
G = nx.read_gml(gml_filename)

# 3) (Optional) If nodes have a "type" attribute (e.g., "question" vs "concept"),
#    color them differently. Otherwise, draw all nodes with one style.
question_nodes = [n for n, d in G.nodes(data=True) if d.get("type") == "question"]
concept_nodes  = [n for n, d in G.nodes(data=True) if d.get("type") == "concept"]

# 4) Choose a layout
pos = nx.spring_layout(G, k=1.0, iterations=50)

# 5) Start plotting
plt.figure(figsize=(12, 9))

# Draw question‐type nodes (if they exist)
if question_nodes:
    nx.draw_networkx_nodes(
        G, pos,
        nodelist=question_nodes,
        node_color="skyblue",
        node_size=300,
        label="Question"
    )

# Draw concept‐type nodes (if they exist)
if concept_nodes:
    nx.draw_networkx_nodes(
        G, pos,
        nodelist=concept_nodes,
        node_color="lightgreen",
        node_size=500,
        label="Concept"
    )

# If no "type" attribute, you could simply do:
# nx.draw_networkx_nodes(G, pos, node_color="lightgray", node_size=400)

# 6) Draw all edges (with arrows)
nx.draw_networkx_edges(
    G, pos,
    arrowstyle="-|>",
    arrowsize=8,
    edge_color="gray"
)

# 7) Draw labels (truncate long ones for readability)
labels = {
    n: (n if len(n) < 30 else n[:27] + "…")
    for n in G.nodes()
}
nx.draw_networkx_labels(G, pos, labels, font_size=8)

# 8) Final touches
plt.title("Knowledge Graph (Questions → Concepts)", fontsize=14)
plt.axis("off")
plt.legend(scatterpoints=1)
plt.tight_layout()

# 9) Save as SVG (instead of PNG)
plt.savefig("knowledge_graph.svg", format="svg", dpi=300)
print("Saved knowledge_graph.svg")

# 10) If you still want to pop up a window, uncomment the next line:
# plt.show()
