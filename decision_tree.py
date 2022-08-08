class BaseDecisionTree():
    def __init__(self, args):
        self.post_pruning = args.post_pruning
        self.use_independent_ts = args.use_independent_ts
        self.max_depth = args.max_depth if args.max_depth > 0 else float('inf')
        self.min_samples_split = args.min_samples_split
        self.alpha = args.alpha
        self.min_prop_split = args.min_prop_split
        self._name = None
        self._tree = None

        if self.post_pruning and self.use_independent_ts:
            self.prune_method = 'REP'
        elif self.post_pruning and not self.use_independent_ts:
            self.prune_method = 'PEP'
        else:
            self.prune_method = ''


    @property
    def tree(self):
        return self._tree


