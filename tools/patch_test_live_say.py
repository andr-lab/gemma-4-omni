import os
import sys

def patch_file(path: str) -> bool:
    with open(path, "r", encoding="utf-8") as f:
        content = f.read()

    old_block = """    assert stream.published_qwen_ids == [17, 16, 15]
    assert stream.source_frontier_snapshot().frozen_source_sequence == 0
    assert stream.take_trailing_token(timeout_s=0.1, q1_index=1) == 16
    reserved = stream._reserved_consumption
    assert reserved is not None
    assert reserved.source_sequences == (4,)
    assert torch.equal(reserved.hidden, torch.full((1, 1, 1536), 51.0))"""

    new_block = """    assert stream.published_qwen_ids == [17, 16]
    assert stream.source_frontier_snapshot().frozen_source_sequence == 0
    assert stream.take_trailing_token(timeout_s=0.1, q1_index=1) == 16
    reserved = stream._reserved_consumption
    assert reserved is not None
    assert reserved.source_sequences == (4,)
    assert torch.equal(reserved.hidden, torch.full((1, 1, 1536), 51.0))
    stream.finish(final_text=stream.visible_text)
    assert stream.published_qwen_ids == [17, 16, 15]"""

    is_crlf = "\r\n" in content
    normalized_content = content.replace("\r\n", "\n")
    if old_block not in normalized_content:
        if new_block in normalized_content:
            print(f"ALREADY PATCHED: {path}")
            return True
        print(f"ERROR: old_block not found in {path}")
        return False

    patched_content = normalized_content.replace(old_block, new_block, 1)
    if is_crlf:
        patched_content = patched_content.replace("\n", "\r\n")

    with open(path, "w", encoding="utf-8") as f:
        f.write(patched_content)
    print(f"SUCCESS: patched {path}")
    return True

if __name__ == "__main__":
    target_path = sys.argv[1] if len(sys.argv) > 1 else "tests/kernel/test_controller_guided_live_say.py"
    if not patch_file(target_path):
        sys.exit(1)
