def pytest_configure(config):
    config.addinivalue_line("markers", "slow: long-running validation (Monte Carlo benchmarks); "
                                       "deselect with -m 'not slow'")
