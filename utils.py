import os

class Parameters:
    devices = ["cuda:0"]

    @staticmethod
    def get_device_ints(limit=3):
        assert "cpu" not in Parameters.devices[:limit]
        final = []
        for item in Parameters.devices[:limit]:
            if isinstance(item, int):
                final.append(item)
            elif item.isnumeric():
                final.append(int(item))
            else:
                f, l = item.split(":")
                final.append(int(l))
        return final


def safe_mkdir(path, force_clean=False):
    if os.path.exists(path) and force_clean:
        os.rmdir(path)
    os.makedirs(path, exist_ok=True)
    return