import tarfile, os

BASE_PATH = os.path.join(os.path.dirname(__file__), "data")

with tarfile.open(os.path.join(BASE_PATH, "lmd_matched.tar.gz"), "r:gz") as tar:
    tar.extractall(os.path.join(BASE_PATH, "lmd_matched"))