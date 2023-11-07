"""
Keras dataset specifications.
"""

def get_datasets(name, **data_args):
    if name == 'dummy':
        from .dummy import get_datasets
        data_args = {k:v for k,v in data_args.items() if k not in ['data_dir', 'stage_dir', 'prestaged', 'train_staging_dup_factor']}
        return get_datasets(**data_args)
    elif name == 'cosmo':
        from .cosmo import get_datasets
        return get_datasets(**data_args)
    elif name == 'dummy_dali':
        from .dummy_dali import get_datasets
        data_args = {k:v for k,v in data_args.items() if k not in ['data_dir', 'stage_dir', 'prestaged', 'do_augmentation', 'validation_batch_size', 'train_staging_dup_factor']}
        return get_datasets(**data_args)
    elif name == 'cosmo_dali':
        from .cosmo_dali import get_datasets
        return get_datasets(**data_args)
    else:
        raise ValueError('Dataset %s unknown' % name)
