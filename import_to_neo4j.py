#!/usr/bin/env python3
"""
import_to_neo4j.py

Reads a GML file (“knowledge_graph.gml”), clears any existing Neo4j database,
and re-imports all nodes and relationships.  Each Question node now has a “text”
property, each Concept/Source node has a “name” property, and every ENRICHED_BY
relationship carries its “source” attribute.
"""

import os
from dotenv import load_dotenv
import networkx as nx
from neo4j import GraphDatabase

# -----------------------------------------------------------------------------
# 1) Load environment variables from a local .env file.  Make sure your
#    .env contains exactly:
#
#    NEO4J_URI=bolt://localhost:7687
#    NEO4J_USER=neo4j
#    NEO4J_PWD=FreshGrace@#002
#
#    (Adjust only if your Desktop-side password changes in Neo4j Browser.)
# -----------------------------------------------------------------------------
load_dotenv()

NEO4J_URI  = os.getenv("NEO4J_URI")
NEO4J_USER = os.getenv("NEO4J_USER")
NEO4J_PWD  = os.getenv("NEO4J_PWD")

if not (NEO4J_URI and NEO4J_USER and NEO4J_PWD):
    raise RuntimeError(
        "Please define NEO4J_URI, NEO4J_USER, and NEO4J_PWD in a .env file."
    )

# -----------------------------------------------------------------------------
# 2) Create a Neo4j driver instance.
# -----------------------------------------------------------------------------
driver = GraphDatabase.driver(
    NEO4J_URI,
    auth=(NEO4J_USER, NEO4J_PWD)
)

# -----------------------------------------------------------------------------
# 3) Helper functions for Cypher calls:
#
#    • clear_database(): deletes everything.
#    • create_question_node(): (:Question {id, text})
#    • create_concept_node():  (:Concept  {id, name})
#    • create_source_node():   (:Source   {id, name})
#    • create_relationship():  (Question)-[:ENRICHED_BY {source}]->(Concept)
# -----------------------------------------------------------------------------

def clear_database(tx):
    """Detach-delete all nodes + relationships in the active database."""
    tx.run("MATCH (n) DETACH DELETE n")


def create_question_node(tx, question_id, question_text):
    """
    Creates a Question node with its ID and text.
    If question_text is None or empty, we still create it with an empty string.
    """
    tx.run(
        """
        CREATE (:Question { id: $id, text: $text })
        """,
        id=question_id,
        text=(question_text if question_text is not None else "")
    )


def create_concept_node(tx, concept_id, concept_name):
    """
    Creates a Concept node with its ID and name.
    If concept_name is None or empty, we still create it with an empty string.
    """
    tx.run(
        """
        CREATE (:Concept { id: $id, name: $name })
        """,
        id=concept_id,
        name=(concept_name if concept_name is not None else "")
    )


def create_source_node(tx, source_id, source_name):
    """
    Creates a Source node with its ID and name.
    If source_name is None or empty, we still create it with an empty string.
    """
    tx.run(
        """
        CREATE (:Source { id: $id, name: $name })
        """,
        id=source_id,
        name=(source_name if source_name is not None else "")
    )


def create_relationship(tx, start_id, end_id, source_label):
    """
    Creates an ENRICHED_BY relationship from Question→Concept,
    setting the 'source' property on that relationship.
    If source_label is missing/null, we default to an empty string.
    """
    tx.run(
        """
        MATCH (q:Question { id: $qid })
        MATCH (c:Concept  { id: $cid })
        CREATE (q)-[:ENRICHED_BY { source: $src }]->(c)
        """,
        qid=start_id,
        cid=end_id,
        src=(source_label if source_label is not None else "")
    )


# -----------------------------------------------------------------------------
# 4) Main import logic:
#    - Read the GML file into NetworkX.
#    - Clear any existing data in Neo4j.
#    - Loop over every node in G: if ID starts with “Q” → Question, “C” → Concept,
#      otherwise → Source.  Extract “text” or “name” accordingly.
#    - Loop over every edge: extract “source” from the edge’s data_dict, and
#      create ENRICHED_BY relationships.
# -----------------------------------------------------------------------------

def main():
    # Path to your local GML file (should be in the same directory as this script):
    gml_path = "knowledge_graph.gml"

    print(f"Loading GML graph from '{gml_path}' …")
    G = nx.read_gml(gml_path)

    print("Connecting to Neo4j and clearing any existing data …")
    with driver.session() as session:
        session.write_transaction(clear_database)

    # 4a) CREATE ALL NODES
    print("Importing nodes …")
    with driver.session() as session:
        for node_id, attrs in G.nodes(data=True):
            if node_id.startswith("Q"):
                # QUESTION: attribute key “text” holds the question wording
                question_text = attrs.get("text")
                session.write_transaction(create_question_node, node_id, question_text)

            elif node_id.startswith("C"):
                # CONCEPT: attribute key “name” holds the concept’s name
                concept_name = attrs.get("name")
                session.write_transaction(create_concept_node, node_id, concept_name)

            else:
                # SOURCE (anything not “Q…” or “C…”)
                # attribute key “name” holds the source’s display name
                source_name = attrs.get("name")
                session.write_transaction(create_source_node, node_id, source_name)

    # 4b) CREATE ALL RELATIONSHIPS
    print("Importing relationships …")
    with driver.session() as session:
        for u, v, edge_data in G.edges(data=True):
            # Edges should be from Question→Concept. Extract “source” from edge attrs.
            rel_source = edge_data.get("source")
            session.write_transaction(create_relationship, u, v, rel_source)

    print("All nodes and relationships have been imported.  Done.")


if __name__ == "__main__":
    try:
        main()
    finally:
        driver.close()
