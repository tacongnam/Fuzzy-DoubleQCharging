class Target:
    def __init__(self, location, cluster_id):
        self.location = location
        self.cluster_id = cluster_id
    
    def get_id(self):
        return self.cluster_id