from audiostream.sources.thread import ThreadSource

class SynthSource(ThreadSource):

    def __init__(self, stream, synth, *args, **kwargs):
        ThreadSource.__init__(self, stream, *args, **kwargs)
        self.chunksize = kwargs.get('chunksize', 1024)
        self.synth = synth
        assert(self.channels == 2)
        assert(self.chunksize % 2 == 0)

    def get_bytes(self):
        samples = self.synth.get_samples(self.chunksize // 2)
        # print(samples.shape, samples.dtype, samples[:5])
        return samples.tobytes()