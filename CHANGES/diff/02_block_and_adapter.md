# block.py / hermes_adapter.py — extraction note (r59d)

block.py is a mechanical de-methodisation of these plugin regions:
  _render_cfg     lines 728-757  (verbatim; self._max_chars -> max_chars arg)
  _wrapper_recall lines 509-560  (verbatim; module lookup -> explicit arg)
  render loop     lines 827-880  (verbatim; self.* -> explicit args)
No logic changes: same caps, same item truncation marker, same block cap marker,
same heading format, same order, same '(node)' fallback.

hermes_adapter.py is the only file importing hermes; it holds no memory logic.
The plugin's thinking-trace and cognition blocks are NOT ported (they are
host-specific optional phases); the memory block is what is measured.
