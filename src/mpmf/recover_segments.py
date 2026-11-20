"""Inspect a damaged segment log without changing it.

    python -m mpmf.recover_segments STORE [--copy-aside NEW_PATH]

Default: report only. --copy-aside preserves the entire original image,
including the damage, at a NEW path. Verified suffix deltas may depend on a
missing node or edge: this is an inventory, not a reconstructed usable store.
"""
import argparse
import hashlib
import json
import os
from pathlib import Path
import shutil
import struct

from . import segments
from .format import FileHeader, HEADER_SIZE


def _identity(st):
    return st.st_dev, st.st_ino, st.st_size, st.st_ctime_ns, st.st_mtime_ns


def inspect(source, copy_aside=None):
    source = Path(source)
    # No writer lock (and no new .lock file). Reject a changing source rather
    # than reporting offsets that might come from two different revisions.
    with source.open('rb') as f:
        initial = _identity(os.fstat(f.fileno()))
        image = f.read()
        if initial != _identity(os.fstat(f.fileno())) or initial != _identity(source.stat()):
            raise RuntimeError('source changed during snapshot; retry on a stable copy')
    h = FileHeader.unpack(image[:HEADER_SIZE])
    base = segments.read_base_end(h)
    if base is None or not HEADER_SIZE <= base <= len(image):
        raise ValueError('expected a segmented image with an in-range base_end')
    if hashlib.sha256(memoryview(image)[HEADER_SIZE:base]).digest() != h.checksum:
        raise ValueError('base checksum mismatch; cannot certify a replayable prefix')
    scan = segments.scan(image, base, dim=h.vector_dimensions)
    pos = base
    before = []
    for payload in scan.payloads:

        end = pos + segments.FRAME_SIZE + len(payload)
        before.append([pos, end])
        pos = end
    after = []
    rejected = []
    if scan.status == 'corrupt':
        for start, end in segments.verified_frames_after(image, scan.offset + 1):
            try:
                segments.decode(memoryview(image)[start + segments.FRAME_SIZE:end], h.vector_dimensions)
            except (ValueError, IndexError, struct.error) as exc:
                rejected.append({'offset': start, 'error': str(exc)})
                continue
            after.append([start, end])
    report = dict(status=scan.status, offset=scan.offset, reason=scan.reason, bytes=len(image),
                  sha256=hashlib.sha256(image).hexdigest(), frames_before=before, frames_after=after,
                  undecodable=rejected, copy_aside=None)
    if copy_aside is not None:
        dest = Path(copy_aside)
        if shutil.disk_usage(dest.parent).free < 2 * len(image):
            raise OSError('need 2x image size free before copying')
        # Exclusive creation rejects existing destinations, including aliases of
        # the source. Never follow a symlink, overwrite, or rename over a store.
        fd = os.open(dest, os.O_CREAT | os.O_EXCL | os.O_WRONLY, 0o600)
        with os.fdopen(fd, 'wb') as f:
            f.write(image)
            f.flush()
            os.fsync(f.fileno())
        with dest.open('rb') as f:
            digest = hashlib.file_digest(f, 'sha256').hexdigest() if hasattr(hashlib, 'file_digest') else hashlib.sha256(f.read()).hexdigest()
        if digest != report['sha256']:
            raise OSError(f'copy verification failed: {dest}; original untouched')
        report['copy_aside'] = str(dest)
    return report


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('store')
    mode = ap.add_mutually_exclusive_group()
    mode.add_argument('--dry-run', action='store_true', help='report only (default)')
    mode.add_argument('--copy-aside', metavar='NEW_PATH')
    a = ap.parse_args()
    print(json.dumps(inspect(a.store, a.copy_aside), indent=2))


if __name__ == '__main__':
    main()
