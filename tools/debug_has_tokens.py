import torch
from tests.kernel.test_response_revision import _transaction_context

session, branch, stream, ledger = _transaction_context()
stream.take_initial_token(timeout_s=0.1)
stream.consume_initial_token()

frontier = stream.source_frontier_snapshot()
rows = [{
    "token_id": 18,
    "text_piece": "é",
    "hidden": torch.ones((1, 1, 1536)),
    "source_token_seq": 2,
    "source_group": ((2, 18),),
    "exclusive_kv_cursor": 10,
    "revision_id": "r1",
    "kv_committed": True,
}]

with stream._condition:
    plan = stream.prepare_unconsumed_suffix_locked(
        rows,
        expected_frontier=frontier,
        frozen_prefix_token_ids=(17,),
        revision_id="r1",
        cache_epoch=4,
    )

staged = plan.staged
total_tokens = len(staged._last_qwen_ids or ())
has_tokens = total_tokens > int(plan.preserved_qwen_count)
print("total_tokens:", total_tokens)
print("preserved_qwen_count:", plan.preserved_qwen_count)
print("has_tokens:", has_tokens)
