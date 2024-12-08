import os


if __name__ == "__main__":
    for d in os.listdir('checkpoints'):
        path = os.path.join('checkpoints', d)
        if len(os.listdir(path)) == 0:
            os.rmdir(path)