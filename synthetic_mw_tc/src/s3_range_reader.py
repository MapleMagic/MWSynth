"""
Seekable, caching file-like object over an S3 object, so HDF5/netCDF4
files can be read IN PLACE on S3 without ever landing on local disk.

WHY THIS EXISTS: the limiting resource for growing the training set is
laptop storage, not bandwidth or patience. The 2023-2025 WHEM GOES data
alone took about 50 GB of a 500 GB machine, and TC PRIMED at the scale
that would actually help (the papers use 13k-18k samples) is far larger
than the disk it would have to sit on. But the training exporter does not
need the files -- it needs a handful of variables out of each one.

netCDF4 is HDF5, and HDF5 is a random-access format: metadata lives in
B-trees, and each variable's data lives in identifiable chunks. Given a
file object that can seek, the HDF5 library will read only the bytes it
actually needs. So the whole download step can be replaced with ranged
GETs, and a 30 MB overpass file can yield its 37/89 GHz channels for a
few MB of transfer and nothing written to disk.

WHY NOT s3fs/fsspec: they do exactly this and do it well, but they are
two more dependencies for a project that already has boto3 as a core
requirement and already opens these buckets unsigned. This is about 100
lines and reuses the existing client.

THE CACHE IS NOT OPTIONAL. HDF5 issues many small reads while walking
metadata -- superblock, B-tree nodes, heap, attributes -- and a naive
implementation that issued one GET per read would produce hundreds of
requests per file and be slower than downloading. Reads are served from
aligned blocks (BLOCK_SIZE), so metadata walking mostly hits cache after
the first block, and a chunked variable read pulls a small number of
large blocks.
"""
from __future__ import annotations

import io
from collections import OrderedDict
from typing import Optional

# Block size for ranged reads. Large enough that HDF5 metadata walking
# usually stays inside one or two blocks; small enough that pulling one
# variable out of a large file does not drag in the whole thing. 4 MB is
# a compromise, and the right knob to turn if a season's mining is
# request-bound rather than bandwidth-bound.
# MEASURED, not guessed. Simulating an HDF5-like access pattern against a
# 30 MB overpass file (scattered small metadata reads, then a handful of
# contiguous chunk reads) gives:
#
#     block     requests   fetched      % of file
#     256 KB      29        7.6 MB        24%
#     512 KB      17        8.9 MB        28%
#       1 MB      14       14.7 MB        47%
#       4 MB       6       23.1 MB        73%
#
# 512 KB sits at the knee: it roughly halves the transfer of a 1 MB block
# for three more requests, while 256 KB buys only another 4 percentage
# points for nearly double the requests. Since S3 request latency is tens
# of milliseconds and these are mined in bulk, requests are not free
# either -- this is a balance, not a minimum.
BLOCK_SIZE = 512 * 1024

# Cap on cached blocks per open file, so a large file cannot quietly grow
# into the memory the rest of the pipeline needs. 16 blocks = 64 MB.
# 128 blocks at 512 KB = 64 MB per open file. Only one file is open at a
# time during mining, and this is well inside a 16 GB machine.
# Block size for a RANGED read of a slice, against the 512 KB used when
# streaming a file end to end.
#
# 1 MB, and the reasoning that first produced 128 KB was WRONG. Recording
# it because the error is instructive.
#
# A live log showed a 0.64% crop fetching 9.4 MB in 18 requests, which
# looked like 25x over-fetch against the 0.38 MB of pixels a crop
# contains. I simulated the access as ~18 scattered 48 KB chunks, found
# smaller blocks cut bytes, and shipped 128 KB.
#
# The next run measured 70 requests and 9.1 MB. The BYTES DID NOT DROP --
# only the request count changed, inversely with block size. That can
# only happen if the access is contiguous, and it is:
#
#   the array is row-major and 5424 wide, so a 434-column crop reads 434
#   separate rows 10848 bytes apart. The pixels total 0.38 MB, but they
#   are strided across a CONTIGUOUS SPAN of 4.7 MB.
#
# So ~5 MB is the floor for this geometry and no block size beats it.
# Smaller blocks cannot help; they only add round trips. Re-simulating
# the true pattern -- 434 row-reads of 868 bytes:
#
#     block   requests   fetched   @50ms   @120ms
#     128 KB        37   4.85 MB   2.39 s   4.98 s
#     512 KB        10   5.24 MB   1.08 s   1.78 s
#    1024 KB         5   5.24 MB   0.83 s   1.18 s
#    2048 KB         3   6.29 MB   0.85 s   1.06 s
#
# 1 MB reaches the byte floor in five requests and is fastest at both
# latencies, so it beats the original 512 KB as well -- the change was
# worth making, just in the opposite direction from the one I took.
RANGED_BLOCK_SIZE = 1024 * 1024

MAX_CACHED_BLOCKS = 128

# Below this size, fetch the WHOLE object in one request instead of
# ranging. MEASURED, and it overturns the 0.104 tuning.
#
# That block size was chosen against a SIMULATED HDF5 access pattern which
# assumed the library touches a small share of a file. Real TC PRIMED
# files say otherwise: across 13 of them, 190 MB of 204 MB was fetched --
# **93%** -- in 386 requests. The variables this project reads (two full
# coordinate grids and four channels at swath resolution) simply span most
# of a 13-20 MB file.
#
# So ranging saves ~7% of transfer and costs ~30 round trips per file.
# At 50 ms latency that is 1.4 s of pure latency per file against 0.05 s
# for a single GET -- roughly 14 minutes over one season, and over an hour
# across 2018-2025, for no benefit.
#
# 64 MB keeps peak memory sane: one object per worker, ten workers, is
# well inside a 16 GB machine. Larger objects keep the ranged path, where
# partial reads may genuinely pay.
WHOLE_OBJECT_MAX_BYTES = 64 * 1024 * 1024


class S3RangeReader(io.RawIOBase):
    """Minimal seekable reader over an S3 object using ranged GETs.

    Implements just enough of the file protocol for h5py: read, seek,
    tell, readable, seekable. h5py accepts any Python file-like object
    with those, and drives it with the same access pattern it would use
    on a local file.

    Tracks bytes_fetched against bytes_served so the saving is
    measurable rather than assumed -- see stats().
    """

    def __init__(self, client, bucket: str, key: str, size: Optional[int] = None,
                 prefer_ranged: bool = False):
        # A caller asking for ranged reads wants a SLICE, so it wants
        # small blocks; the module default suits streaming a file whole.
        self._block_size = RANGED_BLOCK_SIZE if prefer_ranged else BLOCK_SIZE
        self._client = client
        self._bucket = bucket
        self._key = key
        self._pos = 0
        self._blocks: OrderedDict = OrderedDict()
        self.bytes_fetched = 0
        self.requests = 0
        self.bytes_served = 0
        if size is None:
            head = client.head_object(Bucket=bucket, Key=key)
            size = int(head["ContentLength"])
        self._size = int(size)
        self._whole = None
        # prefer_ranged is an INTENT flag, and it has to override the size
        # rule. A full-disk ABI band is ~30 MB compressed -- comfortably
        # under WHOLE_OBJECT_MAX_BYTES -- so the size test alone pulled the
        # entire file and silently threw away the point of cropping. The
        # log showed "0.64% of the array, 30.1 MB fetched", which is the
        # two optimizations contradicting each other in one line.
        #
        # Size cannot decide this on its own: what matters is how much of
        # the object the caller intends to read, which only the caller
        # knows. TC-PRIMED reads ~93% of a small file and should grab it
        # whole; a storm crop reads under 1% of a larger one and must
        # range.
        if not prefer_ranged and self._size <= WHOLE_OBJECT_MAX_BYTES:
            # One request, no ranging. See WHOLE_OBJECT_MAX_BYTES.
            try:
                resp = client.get_object(Bucket=bucket, Key=key)
                self._whole = resp["Body"].read()
                self.requests = 1
                self.bytes_fetched = len(self._whole)
            except Exception:
                self._whole = None      # fall back to ranged reads

    # --- file protocol ------------------------------------------------
    def readable(self):
        return True

    def seekable(self):
        return True

    def tell(self):
        return self._pos

    def seek(self, offset, whence=io.SEEK_SET):
        if whence == io.SEEK_SET:
            self._pos = offset
        elif whence == io.SEEK_CUR:
            self._pos += offset
        elif whence == io.SEEK_END:
            self._pos = self._size + offset
        else:
            raise ValueError(f"invalid whence: {whence}")
        self._pos = max(0, min(self._pos, self._size))
        return self._pos

    def read(self, size=-1):
        if size is None or size < 0:
            size = self._size - self._pos
        size = max(0, min(size, self._size - self._pos))
        if size == 0:
            return b""
        out = bytearray()
        remaining, pos = size, self._pos
        while remaining > 0:
            idx = pos // self._block_size
            block = self._block(idx)
            start = pos - idx * self._block_size
            take = min(remaining, len(block) - start)
            if take <= 0:
                break
            out += block[start:start + take]
            pos += take
            remaining -= take
        self._pos = pos
        self.bytes_served += len(out)
        return bytes(out)

    def readinto(self, b):
        data = self.read(len(b))
        b[:len(data)] = data
        return len(data)

    # --- block cache --------------------------------------------------
    def _block(self, idx: int) -> bytes:
        if self._whole is not None:
            start = idx * self._block_size
            return self._whole[start:start + self._block_size]
        cached = self._blocks.get(idx)
        if cached is not None:
            self._blocks.move_to_end(idx)
            return cached
        start = idx * self._block_size
        end = min(start + self._block_size, self._size) - 1
        if end < start:
            return b""
        resp = self._client.get_object(
            Bucket=self._bucket, Key=self._key, Range=f"bytes={start}-{end}"
        )
        data = resp["Body"].read()
        self.requests += 1
        self.bytes_fetched += len(data)
        self._blocks[idx] = data
        while len(self._blocks) > MAX_CACHED_BLOCKS:
            self._blocks.popitem(last=False)
        return data

    # --- diagnostics --------------------------------------------------
    def stats(self) -> dict:
        """Transfer accounting. `fetched_fraction` is the headline: the
        share of the file that actually crossed the network."""
        return {
            "size_bytes": self._size,
            "bytes_fetched": self.bytes_fetched,
            "bytes_served": self.bytes_served,
            "requests": self.requests,
            "fetched_fraction": (self.bytes_fetched / self._size) if self._size else 0.0,
            "mode": "whole" if self._whole is not None else "ranged",
        }


def open_s3_hdf5(client, bucket: str, key: str, size: Optional[int] = None,
                 prefer_ranged: bool = False):
    """Open an S3-hosted HDF5/netCDF4 file for reading, without
    downloading it. Returns (h5py.File, reader); close the file when done
    and read `reader.stats()` for transfer accounting.

    Raises ImportError if h5py is unavailable, which is the same
    condition under which the existing local-file reader fails, so
    callers need no new error handling.
    """
    import h5py

    reader = S3RangeReader(client, bucket, key, size=size,
                           prefer_ranged=prefer_ranged)
    # h5py needs a buffered wrapper: it does many small reads, and
    # BufferedReader coalesces them before they reach the block cache.
    buffered = io.BufferedReader(reader, buffer_size=1024 * 1024)
    return h5py.File(buffered, "r"), reader
