def bpe_expand_safe(src, max_out=1<<20):
    """Gage BPE (multi-block). Returns (bytes, consumed, status); never blows the stack."""
    pos = 0; out = bytearray(); n = len(src)
    while pos < n:
        left = list(range(256)); right = [0]*256; c = 0
        try:
            while c < 256:
                count = src[pos]; pos += 1
                if count > 127: c += count - 127; count = 0
                if c == 256: break
                for _ in range(count + 1):
                    left[c] = src[pos]; pos += 1
                    if c != left[c]: right[c] = src[pos]; pos += 1
                    c += 1
                    if c > 256: return bytes(out), pos, 'badtable'
            size = (src[pos] << 8) | src[pos+1]; pos += 2
        except IndexError:
            return bytes(out), pos, 'trunc'
        stack = []
        while True:
            if stack: ch = stack.pop()
            elif size == 0: break
            else:
                if pos >= n: return bytes(out), pos, 'trunc'
                ch = src[pos]; pos += 1; size -= 1
            if ch == left[ch]: out.append(ch)
            else:
                stack.append(right[ch]); stack.append(left[ch])
                if len(stack) > 512: return bytes(out), pos, 'stack'
            if len(out) > max_out: return bytes(out), pos, 'overflow'
    return bytes(out), pos, 'ok'
