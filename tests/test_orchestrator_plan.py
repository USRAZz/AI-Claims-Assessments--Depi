from pipeline.orchestrator_in import sanitize_plan


def test_valid_plan_is_kept():
    assert sanitize_plan(["policy", "docs"]) == ["policy", "docs"]


def test_policy_worker_is_always_added():
    assert sanitize_plan(["docs"]) == ["policy", "docs"]


def test_duplicates_and_unknown_workers_are_removed():
    assert sanitize_plan(["history", "history", "hacker", "policy"]) == ["history", "policy"]


def test_empty_plan_falls_back_to_all_workers():
    assert sanitize_plan([]) == ["policy", "history", "docs"]


def test_invalid_plan_falls_back_to_all_workers():
    assert sanitize_plan(["nope"]) == ["policy", "history", "docs"]
    assert sanitize_plan(None) == ["policy", "history", "docs"]
