"""Index-only epoch replay for a workers=0 PyTorch batch DataLoader.

No dataset item is fetched while skipping. The original iterator is constructed
before restoring the saved RNG, neutralizing its new base-seed draw. The full
epoch length remains visible to the unmodified MMCV epoch/LR loop.
"""


def require(value, message):
    if not value:
        raise RuntimeError(message)


def validate_cursor(cursor, *, completed_updates, completed_epochs, loader_batches):
    require(type(cursor) is dict and set(cursor) == {
        'schema_version', 'epoch', 'completed_batches', 'loader_batches', 'sampler', 'iterator_created'},
        'Exact epoch cursor schema required')
    for name in ('schema_version', 'epoch', 'completed_batches', 'loader_batches'):
        require(type(cursor[name]) is int, 'Integer epoch cursor field required: ' + name)
    require(cursor['schema_version'] == 1 and cursor['epoch'] == completed_epochs
        and cursor['loader_batches'] == loader_batches and loader_batches > 0,
        'Saved/current epoch or full loader length differs')
    require(0 <= cursor['completed_batches'] <= loader_batches,
        'Saved epoch cursor outside the full original loader')
    require(type(completed_epochs) is int and completed_epochs >= 0
        and type(completed_updates) is int and completed_updates >= 0 and completed_updates ==
        completed_epochs * loader_batches + cursor['completed_batches'],
        'Saved completed update/epoch/cursor arithmetic differs')
    require(type(cursor['sampler']) is dict and cursor['sampler'], 'Saved sampler contract required')
    require(type(cursor['iterator_created']) is bool
        and (cursor['iterator_created'] or cursor['completed_batches'] == 0),
        'Nonzero cursor requires an existing original epoch iterator')
    return cursor['completed_batches']


class CursorBatchSampler:
    def __init__(self, original, completed_batches):
        require(type(completed_batches) is int and 0 <= completed_batches <= len(original),
            'Batch-index cursor outside original batch sampler')
        self.original = original
        self.completed_batches = completed_batches

    def __iter__(self):
        iterator = iter(self.original)
        for _ in range(self.completed_batches):
            try:
                next(iterator)
            except StopIteration:
                raise RuntimeError('Original sampler ended before saved cursor')
        yield from iterator

    def __len__(self):
        return len(self.original) - self.completed_batches


class EpochCursorLoader:
    """Delegate loader attributes/length, replace only the iterator's indices."""
    def __init__(self, original, completed_batches, rng, restore):
        require(original.num_workers == 0, 'Exact epoch replay requires workers=0')
        require(getattr(original, '_auto_collation', False) is True,
            'Reviewed automatic-batch DataLoader required')
        require(original.batch_sampler is not None and not getattr(original, 'persistent_workers', False),
            'Original nonpersistent batch sampler required')
        self.original = original
        self.completed_batches = completed_batches
        self.rng = rng
        self.restore = restore
        self.iterator_created = False

    def __getattr__(self, name):
        return getattr(self.original, name)

    def __len__(self):
        # MMCV computes max_iters and epoch-based warmup from the original full
        # length, even when only a suffix executes after a mid-epoch restore.
        return len(self.original)

    def __iter__(self):
        require(not self.iterator_created, 'An epoch-resume suffix iterator may be constructed only once')
        self.iterator_created = True
        original = self.original.batch_sampler
        replacement = CursorBatchSampler(original, self.completed_batches)
        try:
            # DataLoader normally rejects public batch_sampler assignment. Only
            # the iterator-construction window uses the index-only wrapper; the
            # original DataLoader and all transforms/collation remain unchanged.
            object.__setattr__(self.original, 'batch_sampler', replacement)
            iterator = iter(self.original)
        finally:
            object.__setattr__(self.original, 'batch_sampler', original)
        self.restore(self.rng)
        return iterator


class EpochPaused(RuntimeError):
    def __init__(self, completed_updates):
        super().__init__('Epoch paused after a completed update and complete checkpoint')
        self.completed_updates = completed_updates
