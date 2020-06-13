class listPersonDetected:
    def __init__(self,personID, centroid):
        self.m_personID=personID
        self.m_centroids=[centroid] # list of center location history
        self.counted=False