import numpy as np
from collections import OrderedDict
from scipy.spatial import distance as dist

class personTracker:
    def __init__(self,maxNbrFramesDisappeared=50,maxDistance=50):
        self.m_nextPersonID=0 # new ID is assigned for new person confirmed
        self.m_list_personCentroids=OrderedDict() # list of PersonCentroid ['personID':centroid,.....] with centroid: (x,y)
        self.m_nbrFramesDisappeared=OrderedDict() # list of (nbr of consecutives frames that personID is marked "not found"
                                                  # ['personID': nbr_frames that personID "not found",.......]

        self.m_maxDistance = maxDistance # dist(existing centroid vs detected centroid) > maxDistance
                                         # -> existing person marked as "not found"

        self.m_maxNbrFramesDisappeared = maxNbrFramesDisappeared # remove person if person's m_nbrFramesDisappeared > m_maxNbrFramesDisappeared


    def addNewPerson(self,centroid):
        self.m_list_personCentroids[self.m_nextPersonID]=centroid # add new personID by its centroid (x,y) in frame
        self.m_nbrFramesDisappeared[self.m_nextPersonID]=0
        self.m_nextPersonID+=1

    def removePerson(self, personID):
        del self.m_list_personCentroids[personID]
        del self.m_nbrFramesDisappeared[personID]

    # for each frame:
    # list of (bounding box is defined by : (startX, startY, endX, endY))
    def update(self,list_bbox_detected):
        #print("Update ....")
        # if no bbox detected
        if (len(list_bbox_detected)==0):
            for personID in list(self.m_nbrFramesDisappeared.keys()):
                self.m_nbrFramesDisappeared[personID]+=1

                if(self.m_nbrFramesDisappeared[personID] > self.m_maxNbrFramesDisappeared):
                    self.removePerson(personID)

            return self.m_list_personCentroids
        else:
            # add centroids position in list_centroids
            list_centroids_detected =  np.zeros((len(list_bbox_detected),2),dtype="int")
            for ( i, (startX,startY,endX,endY)) in enumerate(list_bbox_detected):
                cX = int((startX + endX)/2.0)
                cY = int((startY+endY)/2.0)
                list_centroids_detected[i]= (cX,cY)
            #print("list centroids detected: ", list_centroids_detected)
            #print("list centroids existing: ", self.m_list_personCentroids.values())
            # if currently, we are not tracking any objects
            # take list_centroids and add them as new objects
            if(len(self.m_list_personCentroids)==0):
                for centroid in list_centroids_detected:
                    self.addNewPerson(centroid)
            # otherwise, meaning that we are currently tracking existing objects,
            # try to match centroids detected to existing persons
            else:
                list_personID = list(self.m_list_personCentroids.keys())
                list_existing_centroids = list(self.m_list_personCentroids.values())

                # compute the distance btw each pair of existing centroid and centroid detected on the frame
                # each row is the distance btw one existing_centroid with list_centroids
                # nbr of rows = nbr of existing_centroids
                # nbr of cols = nbr of centroids detected on frame
                D = dist.cdist(np.array(list_existing_centroids),list_centroids_detected)

                # (1) In each row: Find the smallest value corresponding to the smallest distance btw
                # the existing-centroid to list_centroids detected on frame
                # (2) then, Sort the row indexes based on the minimum values
                # rows: contains all indexes of list_existing_centroids sorted with
                # the first element of rows is the idx of row containing the smallest value and next, so on
                rows = D.min(axis=1).argsort()

                # With rows already sorted,
                # For each row, Finding the column containing the smallest value
                # and then sorting them based on the ordered rows
                cols = D.argmin(axis=1)[rows]
                # Our goal is to have the index values (row,col) for each existing_centroid

                # keep track of which of the rows and column indexes
                # we have already examined
                usedRows=set()
                usedCols=set()

                # row: idx of existing_centroid
                # col: idx of centroid_detected whose distance to the existing_centroid is smallest
                for (row,col) in zip(rows,cols):
                    if row in usedRows or col in usedCols:
                        continue

                    if D[row,col] >self.m_maxDistance:
                        continue

                    personID = list_personID[row]
                    self.m_list_personCentroids[personID] = list_centroids_detected[col]
                    self.m_nbrFramesDisappeared[personID]=0

                    #indicate that we already examinated (row,col)
                    usedRows.add(row)
                    usedRows.add(col)

                unusedRows = set(range(0, D.shape[0])).difference(usedRows)
                unusedCols = set(range(0, D.shape[1])).difference(usedCols)

                # if the nbr of existing centroids > nbr of centroids detected
                # mean that some existing centroids were disappeared
                if(D.shape[0] >=D.shape[1]):
                    for row in unusedRows:
                        personID=list_personID[row]
                        self.m_nbrFramesDisappeared[personID]+=1

                        if(self.m_nbrFramesDisappeared[personID]>self.m_maxNbrFramesDisappeared):
                            self.removePerson(personID)
                # if the nbr of existing centroids < nbr of centroids detected
                # mean that new persons appeared
                else:
                    for col in unusedCols:
                        self.addNewPerson(list_centroids_detected[col])

        return self.m_list_personCentroids





