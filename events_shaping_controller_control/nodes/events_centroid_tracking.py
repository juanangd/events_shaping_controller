import numpy as np

class EventsCentroidTracker:

	def __init__(self, alpha_ema, feature_desire_pos, use_outlier_detection=True):

		self.current_centroid = np.array([0., 0.])
		self.previous_centroid = np.array([0., 0.])
		self.feature_desire_pos = feature_desire_pos
		self.use_outlier_detection = use_outlier_detection
		self.alpha_ema = alpha_ema

	@staticmethod
	def feat_locs_to_jacobian_(feat_locs):

		jacobian = np.empty((2, 2))

		x_ = feat_locs[0, 0]
		y_ = feat_locs[1, 0]

		jacobian[0, 0] = x_ * y_
		jacobian[0, 1] = - (1 + x_ ** 2)
		jacobian[1, 0] = - (1 + y_ ** 2)
		jacobian[1, 1] = - x_ * y_

		return jacobian

	def get_current_error(self, data):

		data_subset = data[:, 0:2]
		unique_data= np.unique(data_subset, axis=0)
		centroid_extracted = np.mean(unique_data, axis=0)
		current_error = centroid_extracted - self.feature_desire_pos
		self.current_error = current_error * self.alpha_ema + (1-self.alpha_ema) * self.previous_centroid
		self.previous_centroid = self.current_centroid
		return self.current_error


