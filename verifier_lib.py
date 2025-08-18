import os
import statistics
import json
from classifiers.dbod import DistanceBasedKeystrokeFeatureOutlierDetector
from classifiers.ecdf import ECDF
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity


def cosine_similarity_with_padding(s1, s2):
    if len(s1) == 0 and len(s2) == 0:
        return 1
    s1 = np.array(s1)
    s2 = np.array(s2)
    # Determine the length of the longer sequence
    max_len = max(len(s1), len(s2))

    # Pad the shorter sequence with zeros (or you could use np.nan or the mean of the sequence)
    if len(s1) < max_len:
        s1 = np.pad(s1, (0, max_len - len(s1)), "constant")
    if len(s2) < max_len:
        s2 = np.pad(s2, (0, max_len - len(s2)), "constant")

    # Reshape both sequences to be 2D arrays (required for sklearn cosine_similarity)
    s1 = s1.reshape(1, -1)
    s2 = s2.reshape(1, -1)

    # Compute and return the cosine similarity
    similarity = cosine_similarity(s1, s2)[0][0]
    return similarity


class Verify:
    """
    Our implementations of the Similarity (both weighted and unweighted),
    Absolute, Relative, and ITAD verifiers
    """

    def __init__(self, p1, p2, p1_t=10, p2_t=10):
        # p1 and p2 are dictionaries of features
        # keys in the dictionaries would be the feature names
        # feature names mean individual letters for KHT
        # feature names could also mean pair of letters for KIT or diagraphs
        # feature names could also mean pair of sequence of three letters for trigraphs
        # feature names can be extended to any features that we can extract from keystrokes
        self.pattern1 = p1
        self.pattern2 = p2
        self.pattern1threshold = (
            p1_t  # sort of feature selection, based on the availability
        )
        self.pattern2threshold = (
            p2_t  # sort of feature selection, based on the availability
        )
        with open(os.path.join(os.getcwd(), "classifier_config.json"), "r") as f:
            config = json.load(f)
        self.common_features = []
        if config["use_feature_selection"]:
            for feature in self.pattern1.keys():
                if feature in self.pattern2.keys():
                    if (
                        len(self.pattern1[feature]) >= self.pattern1threshold
                        and len(self.pattern2[feature]) >= self.pattern2threshold
                    ):
                        self.common_features.append(feature)
        else:
            self.common_features = set(self.pattern1.keys()).intersection(
                set(self.pattern2.keys())
            )
        if config["print_feature_distribution"]:
            self.write_feature_pattern_distribution("FI_feat_pattern_dist.txt")
        if config["use_outlier_detection"]:
            outlier_detector = DistanceBasedKeystrokeFeatureOutlierDetector(
                self.common_features, p1, p2
            )
            self.pattern1, self.pattern2 = outlier_detector.find_inliers()
