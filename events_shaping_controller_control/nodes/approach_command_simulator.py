import os
import time
from pathlib import Path

import cv2 as cv
import kornia
import matplotlib.pyplot as plt
import numpy as np
import torch

from image_warped_events.image_warped_events_evaluator import ImageWarpedEventsEvaluator


class ApproachCommandSimulator:

    def __init__(self, **kwargs):

        self.parameters = kwargs
        self.initial_depth = kwargs.get("initial_depth", 0.25)
        self.initial_vz = kwargs.get("initial_vz", -0.15)
        self.learning_rate = kwargs.get("learning_rate", 50.0)
        self.delta_t = kwargs.get("delta_t", 0.05)
        self.divergence_target = kwargs.get("divergence_target", -1.0)
        self.frame_size = kwargs.get("frame_size", (1024, 768))
        self.sampling_rate = kwargs.get("sampling_rate", 1e4)
        self.focal_length = kwargs.get("focal_length", 80.0)
        self.export_gradient_plots = kwargs.get("export_gradient_plots", False)
        self.export_profile_plots = kwargs.get("export_profile_plots", True)
        self.export_data = kwargs.get("export_data", True)
        self.data_path = kwargs.get("data_path", "")
        self.experiment_name = kwargs.get("experiment_name", "experimentX")
        self.event_threshold_trigger = kwargs.get("event_threshold_trigger", 1.)
        self.distance_stop_criteria = kwargs.get("distance_stop", 0.3)
        self.threshold_event_to_analyze = kwargs.get("threshold_event_to_analyze", 300)
        self.export_image = kwargs.get("export_image", False)
        self.allow_velocity_amp = kwargs.get("allow_velocity_amp", False)

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self.path_to_export_data = Path(self.data_path) / self.experiment_name
        if self.export_data or self.export_profile_plots or self.export_gradient_plots:
            if not self.path_to_export_data.exists():
                os.makedirs(self.path_to_export_data)

        camera_intrinsics = torch.tensor(
            [[self.focal_length, 0., self.frame_size[0] / 2],
             [0., self.focal_length, self.frame_size[1] / 2],
             [0., 0., 1.]], dtype=torch.float64)

        camera_intrinsics_inv = torch.linalg.inv(camera_intrinsics)

        self.iwe_eval = ImageWarpedEventsEvaluator(camera_intrinsics, camera_intrinsics_inv, torch.Size([self.frame_size[1], self.frame_size[0]]), 11, torch.Tensor([2., 2.]), motion_model="translation_divergence", sharpness_function_type="poisson")

        self.init_image = None
        self.accumulated_scale_factor = 1.
        self.step_id = 0.
        self.current_velocity = self.initial_vz
        self.current_distance = self.initial_depth
        self.current_time = 0.0
        self.prev_image = None
        self.current_image = None

        self.all_time_stamps = []
        self.all_velocities = []
        self.all_distance = []

    def initialize_img(self, image_path):
        if self.init_image is not None:
            print(f"There was already an image set")
            return None

        img = cv.imread(image_path, cv.IMREAD_GRAYSCALE)
        img_height, img_width = img.shape
        fixed_width, fixed_height = self.frame_size
        if img_width > fixed_width or img_height > fixed_height:
            # Crop the image from the center
            left = (img_width - fixed_width) // 2
            top = (img_height - fixed_height) // 2
            right = left + fixed_width
            bottom = top + fixed_height
            self.init_image = img[top:bottom, left:right]
        else:
            # Create a new white background image of fixed size
            new_img = np.full((fixed_height, fixed_width), 255, dtype=np.uint8)

            # Calculate position to paste the image in the center
            paste_x = (fixed_width - img_width) // 2
            paste_y = (fixed_height - img_height) // 2

            # Paste the original image onto the white background
            new_img[paste_y:paste_y + img_height, paste_x:paste_x + img_width] = img
            self.init_image = new_img

        self.init_image = torch.tensor(self.init_image, device=self.device, dtype=torch.float32)

    @staticmethod
    def apply_centered_expand_transformation(image, scale_factor, output_size, device):
        img_height, img_width = image.shape[:2]
        center_x, center_y = img_width // 2, img_height // 2

        T1 = torch.tensor([[1, 0, -center_x], [0, 1, -center_y], [0, 0, 1]], device=device, dtype=torch.float32)
        S = torch.tensor([[scale_factor, 0, 0], [0, scale_factor, 0], [0, 0, 1]], device=device, dtype=torch.float32)
        T2 = torch.tensor([[1, 0, center_x], [0, 1, center_y], [0, 0, 1]], device=device, dtype=torch.float32)
        M = torch.mm(torch.mm(T2, S), T1)
        M_affine = M[:2, :]

        image_out = kornia.geometry.warp_affine(image.unsqueeze(0).unsqueeze(0), M_affine.unsqueeze(0), (img_height, img_width), mode="nearest", align_corners=True)

        return image_out.squeeze()

    def optimizize_step(self, events):

        if self.current_distance <= self.distance_stop_criteria or self.current_velocity > 0:
            return True
        self.all_velocities.append(self.current_velocity)
        self.all_distance.append(self.current_distance)
        self.all_time_stamps.append(self.current_time)
        if events is None:
            return False
        der_at_target = 0.
        target_tensor = torch.Tensor([self.divergence_target]).to(torch.float64)

        if events.shape[0] > self.threshold_event_to_analyze and events is not None:
            if self.export_gradient_plots:
                print("No need to plot gradients")
                d_queries = np.linspace(-10, 10, 50)
                all_jacob = []
                all_loss = []
                for d_query in d_queries:
                    parameter = torch.Tensor([d_query]).to(torch.float64)
                    all_jacob.append(self.iwe_eval.jacobian_loss_fn(parameter, events)[0])
                    all_loss.append(self.iwe_eval.loss_fn(parameter, events))

                all_jacob = np.array(all_jacob)
                all_loss = np.array(all_loss)
                min_value_loss_idx = np.argmin(all_loss)
                minimum_glob_d = d_queries[min_value_loss_idx]
                plt.figure()
                plt.subplot(1, 2, 1)
                plt.title("Loss function")
                plt.xlabel("Divergence [1/s]")
                plt.ylabel("Loss function")
                plt.plot(d_queries, all_loss, label="loss")
                plt.plot([self.divergence_target, self.divergence_target], [np.min(all_loss), np.max(all_loss)],label="target")
                plt.legend()
                plt.grid()

                plt.subplot(1, 2, 2)
                plt.title("derivative w.r.t. Divergence")
                plt.xlabel("Divergence [1/s]")
                plt.ylabel("Derivative")
                plt.plot(d_queries, all_jacob, label="derivative")
                plt.plot([self.divergence_target, self.divergence_target], [np.min(all_jacob), np.max(all_jacob)], label="target")
                plt.legend()
                plt.grid()
                plt.close()
                plt.savefig(self.path_to_export_data / f"loss_and_jac_profile_{self.step_id}.png")

            target_tensor = torch.Tensor([self.divergence_target]).to(torch.float64)
            der_at_target = self.iwe_eval.jacobian_loss_fn(target_tensor, events)
            if not self.allow_velocity_amp:
                der_at_target = der_at_target if der_at_target >= 0. else 0.

        self.current_velocity += float(self.learning_rate * der_at_target * np.abs(self.current_velocity))

        return False

    def end_algorithm(self):

        all_vel_array = np.array(self.all_velocities)
        all_distance_array = np.array(self.all_distance)
        all_timestamp_aray = np.array(self.all_time_stamps)
        all_divergence_array = all_vel_array / all_distance_array
        if self.export_profile_plots:

            plt.figure()
            plt.subplot(3, 1, 1)
            plt.plot(all_timestamp_aray, all_divergence_array)
            plt.title("Divergence vs time")
            plt.ylabel("Divergence [1/s]")
            plt.xlabel("Time[s]")
            plt.grid()

            plt.subplot(3, 1, 2)
            plt.plot(all_timestamp_aray, all_distance_array)
            plt.title("Distance vs time")
            plt.ylabel("Distance [m/s]")
            plt.xlabel("Time[s]")
            plt.grid()

            plt.subplot(3, 1, 3)
            plt.plot(all_timestamp_aray, all_vel_array)
            plt.title("Velocity vs time")
            plt.ylabel("Velocity [m/s]")
            plt.xlabel("Time[s]")
            plt.grid()

            plt.savefig(self.path_to_export_data / f"profile_{self.experiment_name}.png")
            plt.close()

        if self.export_data:

            data_to_export = {
                "parameters": self.parameters,
                "distance_array":  all_distance_array,
                "velocity_array": all_vel_array,
                "timestamp_array": all_timestamp_aray
            }

            np.save(self.path_to_export_data / f"data_{self.experiment_name}", data_to_export, allow_pickle=True)


        div_error = np.mean(np.abs(self.divergence_target - all_divergence_array))
        return div_error

    def render_step(self):

        if self.init_image is None:
            print("The algorithm is not initialized with an image")
            return

        iterations = int(self.sampling_rate * self.delta_t)
        period_sampling = 1 / self.sampling_rate
        d0 = self.current_distance
        init_image = self.init_image

        all_events = []
        last_scale_factor = 1.

        for _ in range(iterations):
            self.current_distance += self.current_velocity * period_sampling
            self.current_time += period_sampling

            scale_factor = d0 / self.current_distance
            last_scale_factor = scale_factor
            scale_factor_global = self.accumulated_scale_factor * scale_factor
            self.current_image = self.apply_centered_expand_transformation(init_image, scale_factor_global, self.frame_size, self.device)

            if self.prev_image is None:
                self.prev_image = self.current_image.clone()
            else:
                img_diff = self.current_image - self.prev_image
                positive_events_pixels = (img_diff > self.event_threshold_trigger).nonzero(as_tuple=False)
                if positive_events_pixels.shape[0] > 0:
                    all_events.append(torch.cat((positive_events_pixels.flip(1),
                                                 torch.full((positive_events_pixels.shape[0], 1), self.current_time, device=self.device),
                                                 torch.full((positive_events_pixels.shape[0], 1), 1., device=self.device)), dim=1))

                negative_events_pixels = (img_diff < -self.event_threshold_trigger).nonzero(as_tuple=False)
                if negative_events_pixels.shape[0] > 0:
                    all_events.append(torch.cat((negative_events_pixels.flip(1),
                                                 torch.full((negative_events_pixels.shape[0], 1), self.current_time, device=self.device),
                                                 torch.full((negative_events_pixels.shape[0], 1), -1., device=self.device)), dim=1))
                self.prev_image = self.current_image.clone()
        self.accumulated_scale_factor *= last_scale_factor
        self.step_id += 1
        if self.export_image:
            plt.figure()
            plt.imshow(self.current_image.cpu().numpy())
            plt.savefig(self.path_to_export_data / f"img_step_{self.step_id}.png")
            plt.close()

        if all_events:
            all_events_tensor = torch.cat(all_events, dim=0)
            return all_events_tensor.to(torch.float64)
        return None



cfg = {
    "initial_depth": 1,
    "experiment_name": "simulator_image",
    "initial_vz": -5,
    "delta_t": 0.02,
    "sampling_rate": 1e4,
    "image_size": (1024, 768),
    "export_gradient_plots": False,
    "allow_velocity_amp": True
}

from plots.plot_consistenter import PlotConfig

plt_config = PlotConfig()
plt_config.set_single_column_size()

simulator = ApproachCommandSimulator(**cfg)


simulator.initialize_img("/home/juan/Desktop/Alabama-Chanin-1inch-Polka-Dots.jpg")
events= simulator.render_step()
t_surface = torch.zeros((simulator.frame_size[1], simulator.frame_size[0]), device=simulator.device).to(torch.float64)
t_surface.index_put_((events[:, 1].long(), events[:, 0].long()), events[:, 2], accumulate=False)
t_surface_np_array = t_surface.cpu().numpy()


from plots.plot_consistenter_thesis import PlotConfig

plt_configurer = PlotConfig()

plt.figure()

plt_configurer.set_size(5.25*0.7, 5.25*0.7 * 2.7/4)
img = plt.imshow(t_surface_np_array, cmap='inferno')

cbar = plt.colorbar(img, fraction=0.035, pad=0.04)
cbar.ax.yaxis.set_label_position('right')
cbar.set_label('Events Time Stamp [s]')
plt.xlabel("X [pix]")
plt.ylabel("Y [pix]")
plt.grid(False)
plt.savefig("time_surface_simulator.png")


"""divergence_targets = [-0.7, -1.0, -1.3]
initial_distance = [5., 3., 2.]

for dist in initial_distance:
    for div in divergence_targets:
        learning_rates = list(np.linspace(0.2, 3, 29))

        config = {}
        for id, learning_rate in enumerate(learning_rates):
            name_case = f"test_{id}_lr_{learning_rate}"
            standard_params = {
                "data_path": f"/home/juan/git/catkin_ws/src/events_shaping_controller/events_shaping_controller_control/nodes/approach_data_wth_v_amp_init_depth_{dist}_/approach_command_experiments_{div}",
                "initial_depth": dist,
                "divergence_target": div,
                "experiment_name": name_case,
                "initial_vz": -0.8,
                "learning_rate": learning_rate,
                "delta_t": 0.02,
                "sampling_rate": 1e4,
                "image_size": (1024, 768),
                "export_gradient_plots": False,
                "allow_velocity_amp": True
            }
            config[name_case] = standard_params


        all_div_error = []
        all_lr = []
        last_image = None
        for key, cfg in config.items():

            print(f"running the case name: {key}")
            simulator = ApproachCommandSimulator(**cfg)
            simulator.initialize_img("/home/juan/Desktop/Alabama-Chanin-1inch-Polka-Dots.jpg")
            step_i = 0
            while(True):
                t_start = time.time()
                events = simulator.render_step()
                events_cpu = None
                if events is not None:
                    if events.shape[0] // 3000 > 0 :
                        events[:, 2] = events[:, 2] - events[0, 2]
                        events = events[::events.shape[0] // 3000, :]
                    events_cpu = events.to('cpu')
                end_criteria_met = simulator.optimizize_step(events_cpu)
                t_end = time.time()
                # print(f"iteration #{step_i}: {t_end - t_start}s")
                step_i +=1
                if end_criteria_met is True:
                    break
                last_image = simulator.current_image
            div_error = simulator.end_algorithm()
            all_lr.append(cfg["learning_rate"])
            all_div_error.append(div_error)

        path_ = config[list(config.keys())[0]]["data_path"]
        plt.figure()
        plt.plot(all_lr, all_div_error)
        plt.savefig(f"{path_}/div_vs_error.png")
        plt.close()"""