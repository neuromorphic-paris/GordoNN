class AERDATAfile(object):
    address = []
    timestamp = []
    closest_center = []

    def __init__(self, addresses = [], timestamps = [], closest_center = []):
        self.addresses = addresses
        self.timestamps = timestamps
        self.closest_center = closest_center