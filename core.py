"""Manifest, integrity checks and bounded, selective network reads."""
from pathlib import Path
import csv
import hashlib
import io
import json
import time
import urllib.request
import urllib.error
import struct
import zlib
import binascii
import zipfile
import threading

ROOT = Path(__file__).resolve().parent
MAX_FILE = 512 * 1024**2
SOURCE_LIMIT = 19_000_000_000
_lock = threading.Lock()

def read_json(path):
    return json.loads(Path(path).read_text(encoding='utf-8'))

def write_json(path, obj):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_suffix(path.suffix + '.tmp')
    temp.write_text(json.dumps(obj, indent=2, ensure_ascii=False) + '\n', encoding='utf-8')
    temp.replace(path)

def models():
    return read_json(ROOT / 'metadata/models.json')

def save_models(rows):
    write_json(ROOT / 'metadata/models.json', rows)
    keys = list(dict.fromkeys(k for r in rows for k in r))
    with (ROOT / 'metadata/models.csv').open('w', newline='', encoding='utf-8-sig') as f:
        w = csv.DictWriter(f, fieldnames=keys)
        w.writeheader()
        for r in rows:
            w.writerow({k: json.dumps(v, ensure_ascii=False) if isinstance(v, (list, dict)) else v for k, v in r.items()})

def sha256(path):
    with Path(path).open('rb') as f:
        return hashlib.file_digest(f, 'sha256').hexdigest()

def tensor_hash(state):
    """Canonical sorted names, dtype, shape, and raw tensor bytes; includes BN buffers."""
    import torch
    h = hashlib.sha256()
    for k, v in sorted(state.items()):
        if not isinstance(v, torch.Tensor):
            raise TypeError(f'Non-tensor state entry: {k}')
        t = v.detach().cpu().contiguous()
        header = json.dumps([k, str(t.dtype), list(t.shape)], separators=(',', ':')).encode()
        h.update(len(header).to_bytes(8, 'big'))
        h.update(header)
        h.update(t.reshape(-1).view(torch.uint8).numpy().tobytes())
    return h.hexdigest()

def fetch(url, source, start=None, length=None, max_bytes=MAX_FILE):
    """Never falls back to reading a full archive when HTTP Range is ignored."""
    if length is not None and (length <= 0 or length > max_bytes):
        raise ValueError('Invalid or oversized range')
    reserve = length if length is not None else max_bytes
    ledger_path = ROOT / 'metadata/transfer_ledger.json'
    with _lock:
        ledger = read_json(ledger_path) if ledger_path.exists() else {}
        if ledger.get(source, 0) + reserve > SOURCE_LIMIT:
            raise RuntimeError(f'Source transfer budget reached: {source}')
        ledger[source] = ledger.get(source, 0) + reserve
        write_json(ledger_path, ledger)
    # Reserve worst-case bytes; only return unused reservation after success.
    headers = {'User-Agent': 'CIFAR10-Research-Zoo/1.0', 'Accept-Encoding': 'identity'}
    if start is not None:
        headers['Range'] = f'bytes={start}-{start + length - 1}'
        url += ('&' if '?' in url else '?') + f'zoo_range={start}_{length}'
    req = urllib.request.Request(url, headers=headers)
    for attempt in range(5):
        try:
            response = urllib.request.urlopen(req, timeout=90)
            break
        except (urllib.error.HTTPError, urllib.error.URLError, ConnectionResetError, TimeoutError, OSError) as e:
            if isinstance(e, urllib.error.HTTPError) and e.code not in (429, 500, 502, 503, 504):
                raise
            if attempt == 4:
                raise
            time.sleep(min(30, 5 * 2**attempt))
    with response as r:
        if start is not None:
            expected = f'bytes {start}-{start + length - 1}/'
            if r.status != 206 or not r.headers.get('Content-Range', '').startswith(expected):
                raise RuntimeError('Selective download refused: HTTP 206 / Content-Range mismatch')
        declared = r.headers.get('Content-Length')
        if declared and int(declared) > max_bytes:
            raise RuntimeError('Response exceeds per-file limit; body not read')
        data = r.read((length if length else max_bytes) + 1)
    if len(data) > max_bytes or (length is not None and len(data) != length):
        raise RuntimeError('Response length mismatch')
    with _lock:
        ledger = read_json(ledger_path)
        ledger[source] -= reserve - len(data)
        write_json(ledger_path, ledger)
    return data

class RemoteZip(io.RawIOBase):
    def __init__(self, url, size, source):
        self.url, self.size, self.source, self.pos = url, size, source, 0
    def seekable(self):
        return True
    def tell(self):
        return self.pos
    def seek(self, offset, whence=0):
        self.pos = offset if whence == 0 else self.pos + offset if whence == 1 else self.size + offset
        if self.pos < 0:
            raise ValueError('Negative seek')
        return self.pos
    def read(self, n=-1):
        n = min(n if n >= 0 else self.size - self.pos, self.size - self.pos)
        if n == 0:
            return b''
        data = fetch(self.url, self.source, self.pos, n)
        self.pos += n
        return data

def zip_member_range(url, source, member, archive_size):
    """One bounded request using a saved central-directory index, plus ZIP CRC check."""
    offset, compressed = member['offset'], member['compressed']
    length = min(30 + len(member['name'].encode('utf-8')) + 65535 + compressed, archive_size-offset)
    blob = fetch(url, source, offset, length)
    header = struct.unpack('<4s5H3I2H', blob[:30])
    signature, version, flags, method, mtime, mdate, crc, csize, usize, nlen, xlen = header
    if signature != b'PK\x03\x04' or flags & 1:
        raise ValueError('Invalid/encrypted ZIP member')
    name = blob[30:30+nlen].decode('utf-8' if flags & 2048 else 'cp437')
    if name != member['name']:
        raise ValueError('ZIP member name mismatch')
    payload = blob[30+nlen+xlen:30+nlen+xlen+compressed]
    if member['size'] > MAX_FILE:
        raise ValueError('ZIP member size cap')
    if method == 8:
        decompressor = zlib.decompressobj(-15)
        data = decompressor.decompress(payload, member['size']+1)
        if not decompressor.eof or decompressor.unconsumed_tail:
            raise ValueError('ZIP stream exceeds declared size or is truncated')
    elif method == 0:
        data = payload
    else:
        raise ValueError('Unsupported ZIP compression')
    if len(data) != member['size'] or crc and binascii.crc32(data) != crc:
        raise ValueError('ZIP size/CRC mismatch')
    return data

def download_one(row):
    path = ROOT / row['local_checkpoint_path']
    if path.exists():
        if not row.get('sha256') or sha256(path) != row['sha256']:
            raise RuntimeError(f'Existing unverified/mismatched checkpoint: {path}')
        return row
    spec = row['download_spec']
    if spec['kind'] == 'remote_zip':
        if 'member_index' in spec:
            data = zip_member_range(row['checkpoint_url'], row['source_id'], spec['member_index'], spec['archive_size'])
        else:
            with zipfile.ZipFile(RemoteZip(row['checkpoint_url'], spec['archive_size'], row['source_id'])) as z:
                info = z.getinfo(spec['member'])
                if info.file_size > MAX_FILE:
                    raise ValueError('ZIP member too large')
                data = z.read(info)
    elif spec['kind'] == 'tar_range':
        data = fetch(row['checkpoint_url'], row['source_id'], spec['offset'], spec['length'])
    else:
        data = fetch(row['checkpoint_url'], row['source_id'])
        if spec['kind'] == 'zip':
            with zipfile.ZipFile(io.BytesIO(data)) as z:
                info = z.getinfo(spec['member'])
                if info.file_size > MAX_FILE:
                    raise ValueError('ZIP member too large')
                data = z.read(info)
    digest = hashlib.sha256(data).hexdigest()
    if row.get('sha256') and digest != row['sha256']:
        raise ValueError('SHA256 mismatch')
    if row.get('upstream_sha256_prefix') and not digest.startswith(row['upstream_sha256_prefix']):
        raise ValueError('Upstream SHA256 filename prefix mismatch')
    if row.get('upstream_sha1') and hashlib.sha1(data).hexdigest() != row['upstream_sha1']:
        raise ValueError('Upstream SHA1 mismatch')
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_suffix('.part')
    temp.write_bytes(data)
    temp.replace(path)
    row.update(sha256=digest, checkpoint_bytes=len(data), download_date=time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime()), status='downloaded')
    return row
