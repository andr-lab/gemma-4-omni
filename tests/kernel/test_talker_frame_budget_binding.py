import inspect

from duplex_kernel.engine import PipelineEngine


def test_stateless_talker_accepts_backchannel_frame_budget():
    signature = inspect.signature(PipelineEngine.talk)
    bound = signature.bind(
        object(),
        "Certo.",
        None,
        n_words=1,
        min_frames=2,
        max_frames=32,
    )

    assert bound.arguments["max_frames"] == 32
