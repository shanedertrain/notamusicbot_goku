"""Shared pytest configuration for the notamusicbot test suite."""


def pytest_configure(config):
    config.addinivalue_line("markers", "rvc: legacy RVC runtime tests")
    config.addinivalue_line("markers", "integration: end-to-end tests over real assets")
    config.addinivalue_line("markers", "gpu: requires a CUDA GPU")
