def pytest_configure(config):
    config.addinivalue_line("markers", "data: needs the private scorer outputs (BRAINDIFF_CKPT, READER_DIR)")
