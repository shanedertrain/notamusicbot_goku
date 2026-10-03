"""Evidence tests: proofs against live legacy assets (not portable unit tests).

Review Round-4 section 7: the migration proof that runs against the real
legacy ``input/users.json`` is runtime-specific evidence, not a portable unit
test — it moved here from ``tests/unit/test_user_migration.py``. Skips
gracefully when the legacy file is absent from the checkout.
"""


def pytest_configure(config):
    config.addinivalue_line(
        "markers", "evidence: proof against real legacy assets in this checkout"
    )
