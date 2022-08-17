import ray.tune.suggest.variant_generator


def generate_variants(*args, **kwargs):
    for _, configs in ray.tune.suggest.variant_generator.generate_variants(*args, **kwargs):
        yield configs
