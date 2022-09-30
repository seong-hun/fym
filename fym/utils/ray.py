import ray.tune.search.variant_generator


def generate_variants(configs, repeat=1):
    return [
        config
        for tune_config in configs
        for _ in range(repeat)
        for _, config in ray.tune.search.variant_generator.generate_variants(
            tune_config
        )
    ]
