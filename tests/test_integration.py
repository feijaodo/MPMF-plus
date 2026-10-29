"""
Integration test: full MPMFMemory lifecycle with real embeddings + retrieval.
Run directly: python3 tests/test_integration.py
"""
import os
import sys
import tempfile
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from mpmf import MPMFMemory, Plane, NodeType


def main():
    with tempfile.TemporaryDirectory() as d:
        path = os.path.join(d, "store.mpmf")

        # ---- Session 1: store memories, link them, save ----
        mem = MPMFMemory(path, vector_dimensions=768)
        print("backend:", mem.embedder.backend, "(probing...)")

        mem.remember(
            "We fixed the docker networking bug by attaching the container "
            "to the host bridge network instead of the default bridge.",
            name="docker_bridge_fix", node_type=NodeType.FACT,
            summary="Docker networking fixed via host bridge network.",
            wiki={"tool": "docker", "domain": "networking"}, importance=0.9,
        )
        mem.remember(
            "The user strongly prefers concise, direct answers with no fluff "
            "and gets annoyed by aspirational status updates.",
            name="user_prefers_concise", node_type=NodeType.FACT,
            summary="User prefers concise, direct answers.",
            wiki={"topic": "preference", "subject": "communication"}, importance=0.8,
        )
        mem.remember(
            "Postgres connection pool exhaustion was solved by raising "
            "max_connections and adding pgbouncer in transaction mode.",
            name="pg_pool_fix", node_type=NodeType.FACT,
            summary="Postgres pool exhaustion fixed with pgbouncer.",
            wiki={"tool": "postgres", "domain": "database"}, importance=0.7,
        )
        # associative link: both are infra fixes
        mem.link("docker_bridge_fix", "pg_pool_fix", Plane.ASSOCIATIVE, "RELATED_TO")
        mem.save()
        print("saved. stats:", mem.stats())

        # ---- Session 2: fresh reload, retrieve ----
        mem2 = MPMFMemory(path, vector_dimensions=768)
        assert mem2.stats()["nodes"] == 3, mem2.stats()
        assert mem2.stats()["edges"] == 1, mem2.stats()

        # direct fact fetch
        fact = mem2.get("docker_bridge_fix")
        assert "host bridge" in fact["text"], fact
        assert fact["wiki"]["tool"] == "docker", fact
        print("\n[get] docker_bridge_fix ->", fact["summary"])

        # semantic recall: query never uses the words "docker" or "bridge"
        res = mem2.recall("how did we resolve the container connectivity problem?")
        print("\n[recall] query: container connectivity problem")
        print("  ->", res)
        top_names = [mem2.mem.get_name(nid) for nid in res.top_nodes]
        print("  top_nodes:", top_names)
        # docker fix should rank in the results
        assert "docker_bridge_fix" in top_names, f"docker fix missing from {top_names}"

        # recall about user preference
        res2 = mem2.recall("what writing style does the user want?")
        top2 = [mem2.mem.get_name(nid) for nid in res2.top_nodes]
        print("\n[recall] query: user writing style")
        print("  top_nodes:", top2)
        assert "user_prefers_concise" in top2, f"pref missing from {top2}"

        # raw text was loaded for top nodes
        for nid in res.top_nodes[:3]:
            assert nid in res.raw_texts
        print("\nALL INTEGRATION ASSERTIONS PASSED ✓")


if __name__ == "__main__":
    main()
