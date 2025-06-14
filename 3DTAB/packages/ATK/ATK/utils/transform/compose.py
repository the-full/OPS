class Compose(object):
    def __init__(self, transforms=[]):
        self.transforms = transforms

    def __call__(self, data_dict):
        for t in self.transforms:
            data_dict = t(data_dict)
        return data_dict
