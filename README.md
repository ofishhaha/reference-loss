# reference-loss
We tried several modifications based on triplet loss.

file TripletLoss.py is the baseline triplet loss we used.
Reference_triplet.py and Reference_quadruplet.py are modification versions guided by a reference rule we prposed.
Triplet_center.py tries to combine triplet loss and center loss.

Most of the cases, the Reference_triplet loss can obtain better performance than baseline and triplet_center loss.

Citation:
Mingyang Yu, Zhigang Chang, Qin Zhou, Shibao Zheng,Tai Pang Wu "Reference-oriented Loss for Person Re-identification",IJCNN 2019,Budapest,Hungary.
