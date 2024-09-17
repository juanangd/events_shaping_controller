from threading import Lock
import rospy
import numpy as np

class EventsBuffering:

	def __init__(self, use_event_time=False, reference_against_t0=True, remove_older_packets=True):

		self.ref_against_t0 = reference_against_t0
		self.use_event_time = use_event_time
		self.remove_older_packets = remove_older_packets
		self.events_packets = []
		self.packets_time_stamp = []
		self.last_event_timestamp = None
		self.locker_callback = Lock()

	def events_callback(self, msg):

		self.locker_callback.acquire()
		events_packet = msg.events
		packet_reshaped = np.array(events_packet).reshape((-1 , 4))
		self.events_packets.append(packet_reshaped)
		self.last_event_timestamp = packet_reshaped[-1, 2]
		self.packets_time_stamp.append(packet_reshaped[0, 2])
		self.locker_callback.release()

	def get_last_event_timestamp(self):

		return self.last_event_timestamp

	def pull_data(self, time_to_collect, remove_old_packets = True):

		if not self.events_packets:
			return None

		if self.use_event_time:
			time_now = self.last_event_timestamp
		else:
			time_now = rospy.Time.now().to_sec()

		data_t0 = time_now - time_to_collect
		time_to_remove = time_now - (time_to_collect*1.5)
		idx_valid = np.argwhere(np.array(self.packets_time_stamp) >= data_t0)
		idx_not_removed = np.argwhere(np.array(self.packets_time_stamp) >= time_to_remove)
		if len(idx_valid)==0:
			return None
		first_idx = idx_valid[0][0]
		data_collected = np.concatenate(self.events_packets[first_idx:], axis=0)
		if remove_old_packets:
			firs_idx_not_removed = idx_not_removed[0][0]
			self.packets_time_stamp = self.packets_time_stamp[firs_idx_not_removed:]
			self.events_packets = self.events_packets[firs_idx_not_removed:]

		if self.ref_against_t0:
			data_collected[:, 2] = data_collected[:, 2] - data_collected[0, 2]

		return data_collected
