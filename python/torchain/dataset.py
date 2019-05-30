from torchain.egs import _GetEgs, SeqTensorGenerator


class Keys:
    def __init__(self, rspec):
        self.generator = SeqTensorGenerator(rspec)

    def __iter__(self):
        return self

    def __next__(self):
        if self.generator.done():
            raise StopIteration()
        k = self.generator.key()
        self.generator.next()
        return k


class Values:
    def __init__(self, rspec):
        self.generator = SeqTensorGenerator(rspec)

    def __iter__(self):
        return self

    def __next__(self):
        if self.generator.done():
            raise StopIteration()
        v = self.generator.tensor()
        self.generator.next()
        return v


class GetEgs(_GetEgs):
    def utt_keys(self):
        return Keys(self.rspec)

    def utt_values(self):
        return Values(self.rspec)

    def utts(self):
        return zip(self.utt_keys(), self.utt_values())

    def utt_chunks(self, lazy=True):
        g = (self.load(k) for k in self.utt_keys())
        if lazy:
            return g
        return list(g)

    def chunks(self, lazy=True):
        from itertools import chain
        g = chain.from_iterable(self.utt_chunks(lazy=lazy))
        if lazy:
            return g
        return list(g)
