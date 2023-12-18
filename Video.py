class Video:
    def __init__(self, N, delta, D, values, sizes, buffer_size):
        self.N = N
        self.delta = delta
        self.values = values
        self.sizes = sizes
        self.buffer_size = buffer_size
        self.D = D
