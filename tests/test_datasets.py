from entropydrift.datasets import GSM8K_REPO, MATH500_REPO


def test_hub_repo_ids_are_fully_qualified():
    """Bare legacy ids like "gsm8k" raise HfUriError on current huggingface_hub."""
    for repo in (GSM8K_REPO, MATH500_REPO):
        assert "/" in repo, f"{repo!r} must be 'namespace/name'"


def test_gsm8k_repo_is_the_canonical_one():
    assert GSM8K_REPO == "openai/gsm8k"
