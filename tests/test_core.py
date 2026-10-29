"""
tests/test_core.py — Core tests for MPMF Python Reference Implementation

Tests the binary format, record structures, and the full storage round-trip.
"""

import os
import sys
import struct
import tempfile
import unittest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from mpmf.format import (
    Plane,
    NodeType,
    NodeRecord,
    EdgeRecord,
    PlaneIndexEntry,
    FileHeader,
    MAGIC,
    HEADER_SIZE,
    NODE_SIZE,
    EDGE_SIZE,
    PLANE_INDEX_SIZE,
    pack_string,
    unpack_string,
    align_offset,
    VECTOR_ALIGNMENT,
)
from mpmf.memory import MemoryManager, StringPool


class TestBinaryFormat(unittest.TestCase):
    """Test fixed-size record packing/unpacking."""

    def test_header_size(self):
        self.assertEqual(len(FileHeader().pack()), HEADER_SIZE)

    def test_header_pack_unpack_roundtrip(self):
        h = FileHeader(vector_dimensions=256)
        h.node_count = 42
        h.edge_count = 7
        h2 = FileHeader.unpack(h.pack())
        self.assertEqual(h2.version_major, 0)
        self.assertEqual(h2.version_minor, 1)
        self.assertEqual(h2.node_count, 42)
        self.assertEqual(h2.edge_count, 7)
        self.assertEqual(h2.vector_dimensions, 256)

    def test_header_magic_validation(self):
        bad_data = b"XXXX" + bytes(HEADER_SIZE - 4)
        with self.assertRaises(ValueError):
            FileHeader.unpack(bad_data)

    def test_node_size(self):
        n = NodeRecord(node_id=1, node_type=NodeType.CONCEPT)
        self.assertEqual(len(n.pack()), NODE_SIZE)

    def test_node_pack_unpack_roundtrip(self):
        n = NodeRecord(
            node_id=42, name_offset=100, node_type=NodeType.ENTITY,
            plane_flags=Plane.ASSOCIATIVE | Plane.CAUSAL, edge_count=3,
            importance=0.7,
        )
        n2 = NodeRecord.unpack(n.pack())
        self.assertEqual(n2.node_id, 42)
        self.assertEqual(n2.name_offset, 100)
        self.assertEqual(n2.node_type, NodeType.ENTITY)
        self.assertEqual(n2.plane_flags, Plane.ASSOCIATIVE | Plane.CAUSAL)
        self.assertEqual(n2.edge_count, 3)
        self.assertAlmostEqual(n2.importance, 0.7, places=5)

    def test_node_null_id_rejected(self):
        with self.assertRaises(ValueError):
            NodeRecord(node_id=0)

    def test_node_in_plane(self):
        n = NodeRecord(node_id=1, plane_flags=Plane.ASSOCIATIVE | Plane.PRIORITY)
        self.assertTrue(n.in_plane(Plane.ASSOCIATIVE))
        self.assertTrue(n.in_plane(Plane.PRIORITY))
        self.assertFalse(n.in_plane(Plane.CAUSAL))
        self.assertFalse(n.in_plane(Plane.TEMPORAL))

    def test_edge_size(self):
        e = EdgeRecord(edge_id=1, source_node_id=1, target_node_id=2)
        self.assertEqual(len(e.pack()), EDGE_SIZE)

    def test_edge_pack_unpack_roundtrip(self):
        e = EdgeRecord(edge_id=7, source_node_id=3, target_node_id=5,
                       plane_id=Plane.CAUSAL, weight=0.9)
        e2 = EdgeRecord.unpack(e.pack())
        self.assertEqual(e2.edge_id, 7)
        self.assertEqual(e2.source_node_id, 3)
        self.assertEqual(e2.target_node_id, 5)
        self.assertEqual(e2.plane_id, Plane.CAUSAL)
        self.assertAlmostEqual(e2.weight, 0.9, places=5)

    def test_plane_index_entry_size(self):
        pie = PlaneIndexEntry(Plane.ASSOCIATIVE, source_node_id=1, edge_id=1)
        self.assertEqual(len(pie.pack()), PLANE_INDEX_SIZE)

    def test_plane_bitmask(self):
        planes = [Plane.ASSOCIATIVE, Plane.CAUSAL, Plane.TEMPORAL, Plane.FUNCTIONAL, Plane.PRIORITY]
        values = [p.value for p in planes]
        self.assertEqual(values, [0x0001, 0x0002, 0x0004, 0x0008, 0x0010])
        for i, v1 in enumerate(values):
            for v2 in values[i + 1:]:
                self.assertEqual(v1 & v2, 0)

    def test_alignment(self):
        for off in [0, 1, 31, 32, 33, 63, 64, 100, 256]:
            aligned = align_offset(off)
            self.assertEqual(aligned % VECTOR_ALIGNMENT, 0)
            self.assertGreaterEqual(aligned, off)


class TestStringPacking(unittest.TestCase):
    def test_pack_unpack_roundtrip(self):
        for s in ["", "hello", "日本語", "hello world this is a test", "🎉🦉"]:
            unpacked, _ = unpack_string(pack_string(s), 0)
            self.assertEqual(unpacked, s)

    def test_string_pool_dedup(self):
        sp = StringPool()
        a = sp.intern("foo")
        b = sp.intern("bar")
        c = sp.intern("foo")
        self.assertEqual(a, c)
        self.assertNotEqual(a, b)
        data = sp.bytes()
        self.assertEqual(StringPool.read(data, a), "foo")
        self.assertEqual(StringPool.read(data, b), "bar")


class TestStorageRoundTrip(unittest.TestCase):
    """Full store -> save -> reload -> read content cycle."""

    def _mk(self, path, dim=8):
        mm = MemoryManager(path, vector_dimensions=dim)
        mm.load()
        return mm

    def test_content_roundtrip(self):
        with tempfile.TemporaryDirectory() as d:
            path = os.path.join(d, "m.mpmf")
            mm = self._mk(path)
            vec = struct.pack("<8f", *range(8))
            n = mm.create_node(
                name="docker_fix", node_type=NodeType.FACT,
                plane_flags=Plane.ASSOCIATIVE.value,
                text="Fixed docker via bridge network.",
                summary="Docker bridge fix.",
                wiki={"tool": "docker", "fix": "bridge network"},
                vector=vec, importance=0.9,
            )
            mm.periodic_save()

            mm2 = self._mk(path)
            self.assertEqual(mm2.header.node_count, 1)
            nid = mm2.get_node_id_by_name("docker_fix")
            self.assertEqual(nid, n.node_id)
            self.assertIn("bridge network", mm2.get_text(nid))
            self.assertIn("Docker bridge", mm2.get_summary(nid))
            self.assertEqual(mm2.get_wiki(nid).get("tool"), "docker")
            self.assertEqual(mm2.get_wiki(nid).get("fix"), "bridge network")
            v = mm2.get_vector(nid)
            self.assertAlmostEqual(v[1], 1.0, places=5)
            self.assertAlmostEqual(mm2.get_node(nid).importance, 0.9, places=5)

    def test_edges_roundtrip(self):
        with tempfile.TemporaryDirectory() as d:
            path = os.path.join(d, "m.mpmf")
            mm = self._mk(path)
            a = mm.create_node("a", NodeType.CONCEPT, Plane.ASSOCIATIVE.value)
            b = mm.create_node("b", NodeType.CONCEPT, Plane.ASSOCIATIVE.value)
            c = mm.create_node("c", NodeType.CONCEPT, Plane.ASSOCIATIVE.value)
            mm.create_edge(a.node_id, b.node_id, Plane.ASSOCIATIVE, "RELATED_TO")
            mm.create_edge(a.node_id, c.node_id, Plane.CAUSAL, "CAUSES")
            mm.periodic_save()

            mm2 = self._mk(path)
            self.assertEqual(mm2.header.edge_count, 2)
            edges = mm2.get_edges_from(a.node_id)
            self.assertEqual(len(edges), 2)
            labels = {mm2.get_edge_label(e.edge_id) for e in edges}
            self.assertEqual(labels, {"RELATED_TO", "CAUSES"})

    def test_plane_filter(self):
        with tempfile.TemporaryDirectory() as d:
            path = os.path.join(d, "m.mpmf")
            mm = self._mk(path)
            mm.create_node("assoc", NodeType.CONCEPT, Plane.ASSOCIATIVE.value)
            mm.create_node("causal", NodeType.CONCEPT, Plane.CAUSAL.value)
            mm.create_node("both", NodeType.CONCEPT,
                           Plane.ASSOCIATIVE.value | Plane.CAUSAL.value)
            mm.periodic_save()
            mm2 = self._mk(path)
            self.assertEqual(len(mm2.nodes_in_plane(Plane.ASSOCIATIVE)), 2)
            self.assertEqual(len(mm2.nodes_in_plane(Plane.CAUSAL)), 2)
            self.assertEqual(len(mm2.nodes_in_plane(Plane.TEMPORAL)), 0)

    def test_checksum_detects_corruption(self):
        with tempfile.TemporaryDirectory() as d:
            path = os.path.join(d, "m.mpmf")
            mm = self._mk(path)
            mm.create_node("x", NodeType.FACT, Plane.ASSOCIATIVE.value, text="hello")
            mm.periodic_save()
            with open(path, "r+b") as f:
                f.seek(HEADER_SIZE + 300)
                f.write(b"\xff\xff\xff\xff")
            with self.assertRaises(ValueError):
                self._mk(path)


if __name__ == "__main__":
    unittest.main()
