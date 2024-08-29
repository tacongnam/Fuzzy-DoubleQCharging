class Target:
    def __init__(self, id, location, cluster_id):
        self.location = location
        self.id = id
        self.cluster_id = cluster_id
    
    def get_id(self):
        return self.cluster_id