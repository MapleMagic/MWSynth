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
# into the memory the rest of the pipeline needs: 128 blocks is 64 MB at
# the 512 KB streaming size and 8 MB at the 64 KB ranged size.
# Block size for a RANGED read of a slice, against the 512 KB used when
# streaming a file end to end.
#
# 64 KB, MEASURED against a live full-disk band in 0.157 -- and the two
# earlier answers (128 KB, then 1 MB) were both reasoned from a storage
# layout nobody had looked at.
#
# 0.156 said: `Rad` is row-major and contiguous, so a 434-column crop is
# 434 rows strided across a 4.7 MB span, ~5 MB is the floor, and 1 MB
# blocks reach it fastest. The real file says otherwise:
#
#     Rad (5424, 5424) int16, CHUNKED 226x226, gzip 1 + shuffle,
#     524 stored chunks, median 56 KB
#
# A 434x434 crop touches 9 chunks. Traced at the HDF5 level, the whole
# crop -- Rad and DQF chunks, coordinates, metadata -- needs 0.58 MB in
# 40 spans. 0.156 fetched 9.4-10.4 MB for it, for two reasons:
#
#   1. open_s3_hdf5 wrapped the reader in a 1 MiB BufferedReader. Every
#      seek discards it, so each small HDF5 read refilled a full MiB.
#      That alone inflated 0.58 MB of need to 6.9 MB of reads.
#   2. Each ~56 KB chunk sits in a different 1 MB block.
#
# The chunks now arrive through prefetch() as exact ranges, in parallel,
# so blocks serve only the ~10 small serial metadata reads HDF5 makes
# (header, coordinates, B-tree nodes). Measured, pixels bit-identical:
#
#     block      fetched   requests
#     old path   10.44 MB        10   (1 MiB buffer + 1 MiB blocks)
#      16 KB      0.76 MB        32
#      64 KB      1.20 MB        29
#     128 KB      1.79 MB        28
#       1 MB     10.97 MB        28
#
# The serial round trips are ~10 in every row -- prefetch requests run
# concurrently -- so smaller blocks buy bytes without buying latency,
# until 16 KB starts splitting metadata reads. 64 KB: 8.7x fewer bytes.
#
# The lesson is the same one TC PRIMED taught in 0.121: open the real
# file before modelling its access pattern.
RANGED_BLOCK_SIZE = 64 * 1024

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
        # Exact byte spans fetched ahead of time by prefetch(): sorted
        # starts, and start -> bytes. Consulted before the block cache.
        self._span_starts: list = []
        self._spans: dict = {}
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
        hit = self._from_spans(self._pos, size)
        if hit is not None:
            self._pos += len(hit)
            self.bytes_served += len(hit)
            return hit
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

    # --- exact-span prefetch -----------------------------------------
    def prefetch(self, spans, max_workers: int = 16) -> int:
        """Fetch exact byte spans IN PARALLEL, ahead of the reads that
        will want them. Returns the number of bytes fetched.

        Why this exists (measured in 0.157 against live S3): a 434x434
        full-disk crop needs 0.58 MB -- nine 226x226 gzip chunks of Rad,
        the matching DQF chunks, coordinates and metadata. Block reads
        fetched 9.4 MB for it, because each ~56 KB chunk sits in a
        different 1 MB block. Smaller blocks cut bytes but HDF5 issues
        its reads SERIALLY, so every chunk became a round trip. Knowing
        the crop bounds, the caller can ask the chunk index for each
        chunk's exact offset and hand them all here at once: exact bytes,
        one parallel wave of requests.

        Overlapping or already-covered spans are skipped. A failed span is
        simply left out; the normal block path serves it on demand.
        """
        if self._whole is not None:
            return 0
        todo = sorted({(int(o), int(n)) for o, n in spans
                       if n and n > 0 and 0 <= o < self._size})
        todo = [(o, min(n, self._size - o)) for o, n in todo
                if self._from_spans(o, n, peek=True) is None]
        if not todo:
            return 0

        def _get(span):
            o, n = span
            try:
                resp = self._client.get_object(
                    Bucket=self._bucket, Key=self._key,
                    Range=f"bytes={o}-{o + n - 1}")
                return o, resp["Body"].read()
            except Exception:
                return o, None

        from concurrent.futures import ThreadPoolExecutor
        import bisect
        got = 0
        with ThreadPoolExecutor(max_workers=max(1, min(max_workers, len(todo)))) as pool:
            for o, data in pool.map(_get, todo):
                if not data:
                    continue
                self.requests += 1
                self.bytes_fetched += len(data)
                got += len(data)
                if o not in self._spans:
                    bisect.insort(self._span_starts, o)
                self._spans[o] = data
        return got

    def _from_spans(self, pos: int, size: int, peek: bool = False):
        """Bytes [pos, pos+size) if ONE prefetched span wholly contains
        them, else None. HDF5 reads a chunk in a single call at its exact
        offset, so the containing span is found by one bisect."""
        if not self._span_starts:
            return None
        import bisect
        i = bisect.bisect_right(self._span_starts, pos) - 1
        if i < 0:
            return None
        start = self._span_starts[i]
        data = self._spans[start]
        if pos + size > start + len(data):
            return None
        if peek:
            return b""
        return data[pos - start:pos - start + size]

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
    if reader._whole is not None:
        # In memory already; the buffer is harmless and saves Python calls.
        return h5py.File(io.BufferedReader(reader, buffer_size=1024 * 1024), "r"), reader
    # RANGED: no BufferedReader. Measured in 0.157, a 1 MiB buffer here
    # was the single largest source of over-fetch: every seek discards it,
    # and the next small HDF5 read -- a 4 KB B-tree node or a 56 KB chunk
    # -- refilled a full MiB from that position. On a real full-disk crop
    # that turned 0.58 MB of need into 6.9 MB of reads BEFORE the block
    # cache even saw them. The block cache already coalesces small reads.
    return h5py.File(reader, "r"), reader
