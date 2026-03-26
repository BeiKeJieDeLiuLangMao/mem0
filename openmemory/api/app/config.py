import os

USER_ID = os.getenv("USER", "default_user")
DEFAULT_APP_ID = "openmemory"

# Neo4j 配置（从 server.py 迁移）
NEO4J_URI = os.environ.get("NEO4J_URI", "bolt://localhost:7687")
NEO4J_USERNAME = os.environ.get("NEO4J_USERNAME", "neo4j")
NEO4J_PASSWORD = os.environ.get("NEO4J_PASSWORD", "mem0password")
NEO4J_DATABASE = os.environ.get("NEO4J_DATABASE", "neo4j")