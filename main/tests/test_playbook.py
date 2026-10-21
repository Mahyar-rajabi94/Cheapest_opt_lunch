from agent.playbook_loader import find_leaks, system_prompt, version
from agent.schema import ACTION_HELP


def test_playbook_has_no_benchmark_identities():
    assert find_leaks() == []


def test_system_prompt_contains_generated_action_menu():
    sp = system_prompt(["cma_es", "lbfgsb"])
    assert "{{ACTION_MENU}}" not in sp
    for k in ACTION_HELP:
        assert f"`{k}`" in sp
    assert "probing" in sp and "stall-recovery" in sp and "cma_es.md" in sp


def test_no_playbook_ablation_is_role_only():
    full, bare = system_prompt(), system_prompt(with_playbook=False)
    assert len(bare) < len(full) / 2 and "# Skills" not in bare


def test_version_is_stable():
    assert version() == version() and len(version()) == 10
