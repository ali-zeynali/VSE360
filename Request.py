class Request:
    def __init__(self, type, segment, start_time, end_time, bitrates):
        self.type = type
        self.segment = segment
        self.start_time = start_time
        self.end_time = end_time
        self.bitrates = self.justify_bitrate_format(bitrates)  # dictionaty of {tile number -> bitrate downloaded}

    def justify_bitrate_format(self, bitrates):
        justified_bitrates = {}
        for d in bitrates:
            justified_bitrates[int(d)] = int(bitrates[d])
        return justified_bitrates

    def __getArray__(self):
        merged_info = {}
        merged_info['type'] = self.type
        merged_info['segment'] = self.segment
        merged_info['start_time'] = self.start_time
        merged_info['end_time'] = self.end_time
        merged_info['bitrates'] = self.bitrates

        return merged_info

    @staticmethod
    def get_all_bitrates(D, bitrates):
        all_bitrates = [0 for _ in range(D)]
        for d in bitrates:
            all_bitrates[d] = bitrates[d]
        return all_bitrates

    @staticmethod
    def apply_replecement(replace, previous_bitrates):
        new_tiles = 0
        for d in replace.bitrates:
            if d not in previous_bitrates:
                new_tiles += 1
            previous_bitrates[d] = replace.bitrates[d]
        return previous_bitrates, new_tiles

    @staticmethod
    def convert_array_to_Request(request_array):
        return Request(request_array['type'], request_array['segment'], request_array['start_time'], request_array['end_time'], request_array['bitrates'])
