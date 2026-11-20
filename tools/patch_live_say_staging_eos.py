import os
import sys

def patch_file(path: str) -> bool:
    with open(path, "r", encoding="utf-8") as f:
        content = f.read()

    old_block_1 = """            staging.feed_piece(
                str(row["text_piece"]),
                hidden,
                token_index=int(packet_index),
                source_token_id=int(row["token_id"]),
                source_token_seq=int(row["source_token_seq"]),
                source_group=tuple(row["source_group"]),
                exclusive_kv_cursor=int(row["exclusive_kv_cursor"]),
                revision_id=str(row["revision_id"]),
                kv_committed=True,
                seal_initial_segment=True,
            )
        final_text = staging.visible_text
        if final_text.strip():
            staging.finish(final_text=final_text)
        if len(staging._published_ids) < preserved_qwen_count:"""

    new_block_1 = """            staging.feed_piece(
                str(row["text_piece"]),
                hidden,
                token_index=int(packet_index),
                source_token_id=int(row["token_id"]),
                source_token_seq=int(row["source_token_seq"]),
                source_group=tuple(row["source_group"]),
                exclusive_kv_cursor=int(row["exclusive_kv_cursor"]),
                revision_id=str(row["revision_id"]),
                kv_committed=True,
                seal_initial_segment=True,
            )
        if len(staging._published_ids) < preserved_qwen_count:"""

    # Can be old block 2 or previous intermediate
    search_patterns_2 = [
        """        with snapshot._condition:
            plan = snapshot.prepare_unconsumed_suffix_locked(
                replacement_tokens,
                expected_frontier=expected_frontier,
                frozen_prefix_token_ids=frozen_prefix_token_ids,
                revision_id=revision_id,
                cache_epoch=cache_epoch,
            )
        return len(plan.staged._published_ids) > int(plan.preserved_qwen_count)""",
        """        with snapshot._condition:
            plan = snapshot.prepare_unconsumed_suffix_locked(
                replacement_tokens,
                expected_frontier=expected_frontier,
                frozen_prefix_token_ids=frozen_prefix_token_ids,
                revision_id=revision_id,
                cache_epoch=cache_epoch,
            )
        total_tokens = len(plan.staged._published_ids) + len(
            plan.staged._last_qwen_mutable_tail_ids or ()
        )
        return total_tokens > int(plan.preserved_qwen_count)"""
    ]

    new_block_2 = """        with snapshot._condition:
            plan = snapshot.prepare_unconsumed_suffix_locked(
                replacement_tokens,
                expected_frontier=expected_frontier,
                frozen_prefix_token_ids=frozen_prefix_token_ids,
                revision_id=revision_id,
                cache_epoch=cache_epoch,
            )
        total_tokens = len(plan.staged._last_qwen_ids or ())
        return total_tokens > int(plan.preserved_qwen_count)"""

    is_crlf = "\r\n" in content
    normalized_content = content.replace("\r\n", "\n")

    if old_block_1 in normalized_content:
        normalized_content = normalized_content.replace(old_block_1, new_block_1, 1)
        print("Patched block 1 in live_say.py")
    elif new_block_1 in normalized_content:
        print("Block 1 already patched in live_say.py")
    else:
        print("WARNING: block 1 not found")

    patched_2 = False
    for pat in search_patterns_2:
        if pat in normalized_content:
            normalized_content = normalized_content.replace(pat, new_block_2, 1)
            print("Patched block 2 in live_say.py")
            patched_2 = True
            break
    if not patched_2:
        if new_block_2 in normalized_content:
            print("Block 2 already patched in live_say.py")
        else:
            print("WARNING: block 2 not found")

    if is_crlf:
        normalized_content = normalized_content.replace("\n", "\r\n")

    with open(path, "w", encoding="utf-8") as f:
        f.write(normalized_content)
    print(f"SUCCESS: updated {path}")
    return True

if __name__ == "__main__":
    target_path = sys.argv[1] if len(sys.argv) > 1 else "duplex_kernel/live_say.py"
    if not patch_file(target_path):
        sys.exit(1)
