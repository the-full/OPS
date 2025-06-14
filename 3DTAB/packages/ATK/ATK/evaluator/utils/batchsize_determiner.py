class BatchSizeDeterminer:
    def __init__(self, rules):
        self.rules = rules

    def __call__(self, **kwargs):
        for condition, batch_size_func in self.rules.items():
            if condition(**kwargs):
                return batch_size_func(**kwargs)


def get_default_batch_size(**kwargs):
    model_name    = kwargs.get('model_name', 'PointNet')
    batch_size_by_model = {
        'PointNet': 500, 
        'PointNet++_SSG': 150,
        'PointNet++_MSG': 50,
        'DGCNN': 100,
    }
    return batch_size_by_model.get(model_name, 150)

rules = {
    (lambda **kwargs: True) : get_default_batch_size
}

bs_determiner = BatchSizeDeterminer(rules)
