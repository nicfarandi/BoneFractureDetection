import pickle

with open("./models/kmeans_sift.pkl", 'rb') as f:
    kmeans_sift_tester = pickle.load(f)

with open("./models/kmeans_akaze.pkl", 'rb') as f:
    kmeans_akaze_tester = pickle.load(f)

with open("./models/rf_model.pkl", 'rb') as f:
    rf_tester = pickle.load(f)