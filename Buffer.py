class Buffer:
    def __init__(self):
        self.buffer_logs = {}
        self.buffer_logs[0] = 0

    def log_buffer(self, time, buffer_value):
        self.buffer_logs[float(time)] = float(buffer_value)

    def get_buffer_value(self, t):
        ordered_buffers = sorted(self.buffer_logs.items(), key=lambda t: t[0])
        for i in range(len(ordered_buffers) - 1):
            if ordered_buffers[i][0] <= t < ordered_buffers[i + 1][0]:
                t1 = ordered_buffers[i][0]
                t2 = ordered_buffers[i + 1][0]
                ratio = (t - t1) / (t2 - t1)
                b1 = ordered_buffers[i][1]
                b2 = ordered_buffers[i + 1][1]
                return float(b1 + ratio * (b2 - b1))
        return ordered_buffers[-1][1]

    @staticmethod
    def get_buffer_from_logs(buffer_logs):
        buffer = Buffer()
        for t in buffer_logs:
            buffer.log_buffer(t,buffer_logs[t])
        return buffer

    def __print__(self):
        for t in self.buffer_logs:
            print("t:{0}, buffer: {1},\t".format(t, self.buffer_logs[t]), end="")
