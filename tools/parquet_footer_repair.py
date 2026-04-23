"""Reconstruct a missing Parquet footer for a crashed writer.

Parquet's ``pyarrow.parquet.ParquetWriter`` appends each row group to disk as
a complete, self-contained unit (column chunks with fully-formed dictionary
and data pages).  The file-level thrift footer — ``FileMetaData`` + 4-byte
length + ``PAR1`` magic — is only written on ``close()``.  If the writer
process is killed, every row group on disk is intact but no reader will
open the file because there is no footer.

This tool walks the footerless file page-by-page (thrift-compact parsing
of every ``PageHeader``), groups pages into column chunks / row groups
using a schema template read from a reference Parquet file that was
produced by the same writer, synthesises a new ``FileMetaData`` pointing
at the actual byte offsets, and appends it to a copy of the file.

Usage
-----

    python tools/parquet_footer_repair.py \
        --broken runs/exp05/phaseA_cadence/.../archive.parquet \
        --template runs/exp05/phaseA_mps/.../archive.parquet \
        --out runs/exp05/phaseA_cadence/.../archive.repaired.parquet

The template must share the writer's schema (identical columns, dtypes,
key/value metadata, codec).  A run produced by the same code path on any
hardware works.

Caveats
-------

* Per-column statistics (min/max/null_count) are **dropped** — the
  rebuilt footer only contains structural metadata.  Readers that rely
  on statistics for predicate pushdown will scan every page instead.
* Only the subset of the Parquet thrift schema we actually need is
  encoded; unknown optional fields in the template footer are **also
  dropped**.  Reading is unaffected.
* Row-group-level ``sorting_columns`` / ``ordinal`` and column-chunk
  ``crypto_metadata`` / bloom filters are not preserved.
"""

from __future__ import annotations

import argparse
import struct
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

MAGIC = b"PAR1"

# --------------------------------------------------------------------------
# Thrift compact types — used in both reader and writer
# --------------------------------------------------------------------------

CT_STOP = 0
CT_BOOL_TRUE = 1
CT_BOOL_FALSE = 2
CT_BYTE = 3
CT_I16 = 4
CT_I32 = 5
CT_I64 = 6
CT_DOUBLE = 7
CT_BINARY = 8
CT_LIST = 9
CT_SET = 10
CT_MAP = 11
CT_STRUCT = 12


def _zigzag(n: int) -> int:
    return (n >> 1) ^ -(n & 1)


def _zigzag_encode(n: int) -> int:
    return (n << 1) ^ (n >> 63)


# ---- thrift reader --------------------------------------------------------


def rd_uvarint(d: bytes, p: int) -> tuple[int, int]:
    r = 0
    s = 0
    while True:
        b = d[p]
        p += 1
        r |= (b & 0x7F) << s
        if (b & 0x80) == 0:
            break
        s += 7
    return r, p


def rd_varint(d: bytes, p: int) -> tuple[int, int]:
    r, p = rd_uvarint(d, p)
    return _zigzag(r), p


def rd_value(d: bytes, p: int, t: int) -> tuple[Any, int]:
    if t == CT_BOOL_TRUE:
        return True, p
    if t == CT_BOOL_FALSE:
        return False, p
    if t == CT_BYTE:
        return d[p], p + 1
    if t in (CT_I16, CT_I32, CT_I64):
        return rd_varint(d, p)
    if t == CT_DOUBLE:
        return struct.unpack_from("<d", d, p)[0], p + 8
    if t == CT_BINARY:
        ln, p = rd_uvarint(d, p)
        return bytes(d[p : p + ln]), p + ln
    if t == CT_STRUCT:
        return rd_struct(d, p)
    if t in (CT_LIST, CT_SET):
        return rd_list(d, p)
    if t == CT_MAP:
        return rd_map(d, p)
    raise ValueError(f"unknown thrift type {t}")


def rd_list(d: bytes, p: int) -> tuple[dict[str, Any], int]:
    hb = d[p]
    p += 1
    size = hb >> 4
    inner = hb & 0x0F
    if size == 0x0F:
        size, p = rd_uvarint(d, p)
    out = []
    for _ in range(size):
        v, p = rd_value(d, p, inner)
        out.append(v)
    return {"_type": "list", "_elem_type": inner, "items": out}, p


def rd_map(d: bytes, p: int) -> tuple[dict[str, Any], int]:
    hb = d[p]
    p += 1
    size = hb >> 4
    if size == 0x0F:
        size, p = rd_uvarint(d, p)
    out: list[tuple[Any, Any]] = []
    kt = vt = 0
    if size > 0:
        kv = d[p]
        p += 1
        kt = kv >> 4
        vt = kv & 0x0F
        for _ in range(size):
            k, p = rd_value(d, p, kt)
            v, p = rd_value(d, p, vt)
            out.append((k, v))
    return {"_type": "map", "_key_type": kt, "_val_type": vt, "items": out}, p


def rd_struct(d: bytes, p: int) -> tuple[dict[int, tuple[int, Any]], int]:
    """Returns {field_id: (thrift_type, value)}."""
    out: dict[int, tuple[int, Any]] = {}
    last_fid = 0
    while True:
        b = d[p]
        p += 1
        if b == 0:
            break
        fd = b >> 4
        ft = b & 0x0F
        if fd == 0:
            fid_raw, p = rd_uvarint(d, p)
            fid = _zigzag(fid_raw)
        else:
            fid = last_fid + fd
        last_fid = fid
        v, p = rd_value(d, p, ft)
        out[fid] = (ft, v)
    return out, p


# ---- thrift writer --------------------------------------------------------


def wr_uvarint(buf: bytearray, n: int) -> None:
    while True:
        b = n & 0x7F
        n >>= 7
        if n == 0:
            buf.append(b)
            return
        buf.append(b | 0x80)


def wr_varint(buf: bytearray, n: int) -> None:
    wr_uvarint(buf, _zigzag_encode(n))


def wr_field_header(buf: bytearray, ft: int, fid: int, last_fid: int) -> int:
    delta = fid - last_fid
    if 1 <= delta <= 15:
        buf.append((delta << 4) | ft)
    else:
        buf.append(ft & 0x0F)
        wr_varint(buf, fid)
    return fid


def wr_value(buf: bytearray, t: int, v: Any) -> None:
    if t in (CT_BOOL_TRUE, CT_BOOL_FALSE):
        return  # no payload (value is encoded in field type)
    if t == CT_BYTE:
        buf.append(v & 0xFF)
        return
    if t in (CT_I16, CT_I32, CT_I64):
        wr_varint(buf, v)
        return
    if t == CT_DOUBLE:
        buf.extend(struct.pack("<d", v))
        return
    if t == CT_BINARY:
        wr_uvarint(buf, len(v))
        buf.extend(v)
        return
    if t == CT_STRUCT:
        wr_struct(buf, v)
        return
    if t in (CT_LIST, CT_SET):
        wr_list(buf, v)
        return
    if t == CT_MAP:
        wr_map(buf, v)
        return
    raise ValueError(f"cannot serialise type {t}")


def wr_list(buf: bytearray, lst: dict[str, Any]) -> None:
    items = lst["items"]
    inner = lst["_elem_type"]
    n = len(items)
    if n < 15:
        buf.append((n << 4) | inner)
    else:
        buf.append(0xF0 | inner)
        wr_uvarint(buf, n)
    for it in items:
        wr_value(buf, inner, it)


def wr_map(buf: bytearray, m: dict[str, Any]) -> None:
    items = m["items"]
    kt = m["_key_type"]
    vt = m["_val_type"]
    n = len(items)
    if n == 0:
        buf.append(0)
        return
    if n < 15:
        buf.append(n << 4)  # size in high nibble, low nibble is 0
    else:
        buf.append(0xF0)
        wr_uvarint(buf, n)
    buf.append((kt << 4) | vt)
    for k, v in items:
        wr_value(buf, kt, k)
        wr_value(buf, vt, v)


def wr_struct(buf: bytearray, fields: dict[int, tuple[int, Any]]) -> None:
    last_fid = 0
    for fid in sorted(fields.keys()):
        ft, v = fields[fid]
        # booleans: field type carries the value
        if ft == CT_BOOL_TRUE or ft == CT_BOOL_FALSE:
            eff_type = CT_BOOL_TRUE if v else CT_BOOL_FALSE
            last_fid = wr_field_header(buf, eff_type, fid, last_fid)
        else:
            last_fid = wr_field_header(buf, ft, fid, last_fid)
            wr_value(buf, ft, v)
    buf.append(0)  # STOP


# --------------------------------------------------------------------------
# Parquet page scanner
# --------------------------------------------------------------------------

# PageType enum
PT_DATA = 0
PT_INDEX = 1
PT_DICT = 2
PT_DATA_V2 = 3


@dataclass
class Page:
    header_start: int  # file offset where PageHeader begins
    header_end: int  # file offset where PageHeader ends (payload begins)
    page_type: int
    uncompressed: int
    compressed: int
    num_values: int  # 0 if not available

    @property
    def data_end(self) -> int:
        return self.header_end + self.compressed


def _extract_num_values(page_fields: dict[int, tuple[int, Any]]) -> int:
    """Extract num_values from a PageHeader by digging into its sub-structs."""
    # DataPageHeader is field 5; DataPageHeaderV2 is field 8; DictionaryPageHeader is field 7
    for fid in (5, 7, 8):
        if fid in page_fields:
            ft, sub = page_fields[fid]
            if ft == CT_STRUCT and isinstance(sub, dict):
                nv_entry = sub.get(1)  # num_values is field 1 in all three
                if nv_entry is not None:
                    return nv_entry[1]
    return 0


def scan_pages(data: bytes) -> list[Page]:
    """Scan every PageHeader in a Parquet file's page region.

    Starts at byte 4 (after the leading ``PAR1`` magic) and walks forward
    until EOF or an unparseable byte.  Each returned :class:`Page`
    records its header start/end and compressed payload size so downstream
    code can group pages into column chunks / row groups.
    """
    pages: list[Page] = []
    p = 4
    while p < len(data):
        try:
            hdr_start = p
            fields, hdr_end = rd_struct(data, p)
        except Exception:
            break
        if 1 not in fields or 2 not in fields or 3 not in fields:
            break
        page_type = fields[1][1]
        uncompressed = fields[2][1]
        compressed = fields[3][1]
        num_values = _extract_num_values(fields)
        pages.append(
            Page(hdr_start, hdr_end, page_type, uncompressed, compressed, num_values)
        )
        p = hdr_end + compressed
        if p > len(data):
            break
    return pages


# --------------------------------------------------------------------------
# Row-group synthesis
# --------------------------------------------------------------------------


@dataclass
class ColumnChunkShape:
    """Template describing one column's chunk layout across row groups.

    Extracted from the reference (template) file's first fully-populated
    row group.  Everything except byte offsets / sizes / num_values is
    reused verbatim.
    """

    path_in_schema: list[bytes]
    type_id: int
    encodings: list[int]
    codec: int
    encoding_stats: Any | None
    has_dictionary: bool


def _extract_col_shapes(template_fm: dict[int, tuple[int, Any]]) -> list[ColumnChunkShape]:
    """Read per-column shape from the template file's first row group."""
    row_groups = template_fm[4][1]["items"]
    # Prefer a row group past index 0 — the first RG of a PDQ run typically
    # has 1 row and may use different encodings than steady-state RGs.
    rg = row_groups[min(1, len(row_groups) - 1)]
    columns = rg[1][1]["items"]
    shapes: list[ColumnChunkShape] = []
    for col in columns:
        meta = col[3][1]  # ColumnChunk.meta_data
        shapes.append(
            ColumnChunkShape(
                path_in_schema=[
                    s for s in meta[3][1]["items"]
                ],
                type_id=meta[1][1],
                encodings=[e for e in meta[2][1]["items"]],
                codec=meta[4][1],
                encoding_stats=meta.get(13, (None, None))[1] if 13 in meta else None,
                has_dictionary=(11 in meta),
            )
        )
    return shapes


@dataclass
class ColumnChunkMeta:
    """Materialised column-chunk metadata for one row group."""

    shape: ColumnChunkShape
    num_values: int
    total_uncompressed: int
    total_compressed: int
    data_page_offset: int
    dictionary_page_offset: int  # -1 if no dict page
    file_offset: int  # where in file this column chunk begins


def _group_pages_into_row_groups(
    pages: list[Page], shapes: list[ColumnChunkShape]
) -> list[list[ColumnChunkMeta]]:
    """Pack the scanned pages into (row_group × column_chunk) structure.

    Assumptions:
      * Columns appear in schema order.
      * Each column chunk has at most one dictionary page, always first.
      * Each column chunk has exactly one data page.
      * Whether a column uses a dictionary page is determined by the
        template (``shape.has_dictionary``).

    These match the writer we're recovering — single ``write_table`` per
    row group via ``pyarrow.parquet.ParquetWriter``.
    """
    n_cols = len(shapes)
    row_groups: list[list[ColumnChunkMeta]] = []
    idx = 0
    while idx < len(pages):
        rg_cols: list[ColumnChunkMeta] = []
        for col_ix, shape in enumerate(shapes):
            if idx >= len(pages):
                raise ValueError(
                    f"ran out of pages mid-row-group; col {col_ix} of RG {len(row_groups)}"
                )
            dict_page: Page | None = None
            # The template says whether this column uses a dictionary.  Trust
            # the scan: if the next page is a dictionary page, consume it.
            if shape.has_dictionary and pages[idx].page_type == PT_DICT:
                dict_page = pages[idx]
                idx += 1
            elif pages[idx].page_type == PT_DICT:
                # Template said no dict but we see one — take it anyway.
                dict_page = pages[idx]
                idx += 1
            if idx >= len(pages):
                raise ValueError(
                    f"missing data page; col {col_ix} of RG {len(row_groups)}"
                )
            data_page = pages[idx]
            if data_page.page_type not in (PT_DATA, PT_DATA_V2):
                raise ValueError(
                    f"expected data page, got type {data_page.page_type} at offset {data_page.header_start}"
                )
            idx += 1

            data_offset = data_page.header_start
            dict_offset = dict_page.header_start if dict_page else -1
            chunk_start = dict_offset if dict_page else data_offset
            compressed = data_page.data_end - chunk_start - (
                data_page.header_end - data_page.header_start
            )
            # total_compressed_size is the sum of all compressed page data +
            # page header bytes within this chunk.
            total_compressed = (data_page.data_end - chunk_start)
            total_uncompressed = data_page.uncompressed + (
                dict_page.uncompressed if dict_page else 0
            )
            rg_cols.append(
                ColumnChunkMeta(
                    shape=shape,
                    num_values=data_page.num_values,
                    total_uncompressed=total_uncompressed,
                    total_compressed=total_compressed,
                    data_page_offset=data_offset,
                    dictionary_page_offset=dict_offset,
                    file_offset=chunk_start,
                )
            )
        row_groups.append(rg_cols)
    return row_groups


# --------------------------------------------------------------------------
# FileMetaData synthesis
# --------------------------------------------------------------------------


def _make_column_chunk(cc: ColumnChunkMeta) -> dict[int, tuple[int, Any]]:
    # ColumnMetaData
    meta: dict[int, tuple[int, Any]] = {}
    meta[1] = (CT_I32, cc.shape.type_id)
    meta[2] = (
        CT_LIST,
        {"_type": "list", "_elem_type": CT_I32, "items": list(cc.shape.encodings)},
    )
    meta[3] = (
        CT_LIST,
        {"_type": "list", "_elem_type": CT_BINARY, "items": list(cc.shape.path_in_schema)},
    )
    meta[4] = (CT_I32, cc.shape.codec)
    meta[5] = (CT_I64, cc.num_values)
    meta[6] = (CT_I64, cc.total_uncompressed)
    meta[7] = (CT_I64, cc.total_compressed)
    meta[9] = (CT_I64, cc.data_page_offset)
    if cc.dictionary_page_offset >= 0:
        meta[11] = (CT_I64, cc.dictionary_page_offset)
    # We intentionally omit statistics (field 12) and encoding_stats
    # (field 13) — the writer's originals are not recoverable without
    # decoding every page, and readers do not require them.

    # ColumnChunk wrapper
    col: dict[int, tuple[int, Any]] = {}
    col[2] = (CT_I64, cc.file_offset)
    col[3] = (CT_STRUCT, meta)
    return col


def _make_row_group(cols: list[ColumnChunkMeta]) -> dict[int, tuple[int, Any]]:
    column_chunks = [_make_column_chunk(c) for c in cols]
    num_rows = cols[0].num_values if cols else 0
    rg_start = cols[0].file_offset if cols else 0
    total_byte_size = sum(c.total_uncompressed for c in cols)
    total_compressed = sum(c.total_compressed for c in cols)

    rg: dict[int, tuple[int, Any]] = {}
    rg[1] = (
        CT_LIST,
        {"_type": "list", "_elem_type": CT_STRUCT, "items": column_chunks},
    )
    rg[2] = (CT_I64, total_byte_size)
    rg[3] = (CT_I64, num_rows)
    rg[5] = (CT_I64, rg_start)
    rg[6] = (CT_I64, total_compressed)
    return rg


def _build_filemetadata(
    template_fm: dict[int, tuple[int, Any]],
    row_groups: list[list[ColumnChunkMeta]],
) -> dict[int, tuple[int, Any]]:
    """Clone template's schema/metadata/created_by; replace row groups."""
    fm: dict[int, tuple[int, Any]] = {}
    # 1: version
    fm[1] = template_fm[1]
    # 2: schema (list of SchemaElement) — copy verbatim
    fm[2] = template_fm[2]
    # 3: num_rows — recompute
    fm[3] = (CT_I64, sum(cols[0].num_values for cols in row_groups))
    # 4: row_groups — the thing we're replacing
    rg_items = [_make_row_group(cols) for cols in row_groups]
    fm[4] = (
        CT_LIST,
        {"_type": "list", "_elem_type": CT_STRUCT, "items": rg_items},
    )
    # 5: key_value_metadata — copy if present
    if 5 in template_fm:
        fm[5] = template_fm[5]
    # 6: created_by — copy if present
    if 6 in template_fm:
        fm[6] = template_fm[6]
    # 7: column_orders — copy if present
    if 7 in template_fm:
        fm[7] = template_fm[7]
    return fm


# --------------------------------------------------------------------------
# Top-level repair
# --------------------------------------------------------------------------


def read_template_filemetadata(template_path: Path) -> dict[int, tuple[int, Any]]:
    with open(template_path, "rb") as f:
        f.seek(-8, 2)
        footer_len = struct.unpack("<I", f.read(4))[0]
        magic = f.read(4)
        if magic != MAGIC:
            raise ValueError(f"{template_path}: bad trailing magic {magic!r}")
        sz = template_path.stat().st_size
        f.seek(sz - 8 - footer_len)
        footer_bytes = f.read(footer_len)
    fm, _ = rd_struct(footer_bytes, 0)
    return fm


def repair(
    broken: Path,
    template: Path,
    out: Path,
) -> dict[str, Any]:
    """Repair ``broken`` using structural info from ``template``.

    Writes a new file at ``out`` that contains the broken file's page
    bytes verbatim plus a freshly-synthesised ``FileMetaData`` footer.
    Returns a small summary dict.
    """
    data = broken.read_bytes()
    if data[:4] != MAGIC:
        raise ValueError(f"{broken}: missing leading PAR1 magic")

    template_fm = read_template_filemetadata(template)
    shapes = _extract_col_shapes(template_fm)

    pages = scan_pages(data)
    if not pages:
        raise ValueError(f"{broken}: no pages parsed")
    last_page_end = pages[-1].data_end
    if last_page_end != len(data):
        # Truncate the output at the last complete page; leftover bytes are
        # a partial write we cannot use.
        data = data[:last_page_end]

    # Drop the empty-stub row group that SeedLogger writes on seed-dir
    # creation: if the first page has 0 uncompressed bytes and the file had
    # more pages afterwards, skip it.  In practice, stubs write a few tiny
    # pages then the ParquetWriter closes the stub and opens a new writer
    # from the first real flush — so the "stub" is an entirely separate
    # file, not a leading row group here.  We therefore do NOT strip
    # anything; if the stub was present, the cadence ParquetWriter's
    # first flush overwrote the whole file.

    row_groups = _group_pages_into_row_groups(pages, shapes)

    fm = _build_filemetadata(template_fm, row_groups)

    # Serialise footer
    footer_buf = bytearray()
    wr_struct(footer_buf, fm)
    footer_len = len(footer_buf)

    # Compose output: [broken data (truncated to last page) | footer | len | PAR1]
    out.write_bytes(bytes(data) + bytes(footer_buf) + struct.pack("<I", footer_len) + MAGIC)

    return {
        "n_pages": len(pages),
        "n_row_groups": len(row_groups),
        "n_rows": sum(cols[0].num_values for cols in row_groups),
        "footer_len": footer_len,
        "file_size_before": len(broken.read_bytes()),
        "file_size_after": out.stat().st_size,
    }


# --------------------------------------------------------------------------
# CLI
# --------------------------------------------------------------------------


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--broken", type=Path, required=True, help="Footerless parquet file")
    ap.add_argument(
        "--template",
        type=Path,
        required=True,
        help="Complete parquet from the same writer (schema template)",
    )
    ap.add_argument("--out", type=Path, required=True, help="Output repaired parquet")
    ap.add_argument(
        "--verify",
        action="store_true",
        help="Open output with pyarrow and print row count",
    )
    args = ap.parse_args(argv)

    info = repair(args.broken, args.template, args.out)
    for k, v in info.items():
        print(f"  {k}: {v}")

    if args.verify:
        import pyarrow.parquet as pq

        pf = pq.ParquetFile(str(args.out))
        print(f"  pyarrow-verified: {pf.num_row_groups} RG, {pf.metadata.num_rows} rows")
        tbl = pf.read()
        print(f"  full read: {tbl.num_rows} rows, {tbl.num_columns} cols")

    return 0


if __name__ == "__main__":
    sys.exit(main())