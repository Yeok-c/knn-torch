# Main class including H_SV function generation and classification with KNN
import os
from datetime import datetime

import matplotlib.pyplot as plt
import torch

# n kinda related to m
# v kinda related to k


class KNN:
    def __init__(
        self,
        noisy=False,
        s_mag=100,  # number of centers of the H_sv function
        n_train=1000,  # number of training data
        n_test=4000,  # number of test data
        k_max=200,  # max number of neighbours asked in all trials, specified because all the distances between points are calculated and sorted up to k_max neighbours during init
        plotting_reso=50,  # resolution of grid for plotting the 'background' colors specifying which areas would be classified into 0 or 1 in the H_sv function
        plot_flag=False,  # whether to plot the results
        seed=None,  # seed for reproducibility, currently unused
        save_subfolder="",  # save location for plotted images
    ):

        self.n = s_mag
        self.n_train = n_train
        self.k_max = k_max
        self.plotting_reso = plotting_reso
        self.plot_flag = plot_flag
        self.save_subfolder = save_subfolder
        if not os.path.exists(f"./results/{self.save_subfolder}"):
            os.mkdir(f"./results/{self.save_subfolder}")

        # print("generating dataset of size of: ", n_train, n_test, "plotting reso: ", plotting_reso)
        self.hsv = self._generate_h(
            100
        )  # h_sv is generated once, same function is used for all k values in same trial
        self.data_train = self._generate_from_hsv(
            n_train, noisy
        )  # training data generated by sampling H_sv
        self.data_test = self._generate_from_hsv(
            n_test, noisy
        )  # training data generated by sampling H_sv
        self.k_nearest_indices = self._calc_distances_to_points(
            self.data_test["X"], self.data_train["X"]
        )  # calculate

    # hsv function
    def _generate_h(self, n):
        data = torch.zeros(n, dtype=torch.float32)
        X = torch.zeros((n, 2), dtype=torch.float32)  # n x 2 tensor
        Y = torch.zeros(n, dtype=torch.int32)  # n x 1 tensor
        data = {"X": X, "Y": Y}
        data["X"] = torch.rand(n, 2)  #  Generate X ~ Uniform [0, 1]
        data["Y"] = torch.randint(0, 2, (n,))  #  Generate Y ~ Uniform [0 or 1]
        return data

    def _generate_from_hsv(self, n, noisy):
        X = torch.rand(n, 2)  # Sample X ~ Uniform [0, 1]
        # Data sample from this H_sv distribution has each point's label is determined by the majority vote of its v-nearest neighbors
        Y = self._classify(
            self._calc_distances_to_points(X, self.hsv["X"]), self.hsv, 3
        )

        # If question specifies it to noisy, then 20% of the data is random ~ Uniform [0, 1]
        if noisy:
            # 0.2 of the data follows random, the rest remains unchanged
            random_indices = torch.rand(n) < 0.2
            Y[random_indices] = torch.randint(0, 2, (random_indices.sum(),)).bool()
        data = {"X": X, "Y": Y}
        return data

    def _plot_generate_grid_hsv(self, reso: int, v: int):
        """
        Generate regions with colors to show if test points on that area would be classified as 0 or 1
        This is done by classifiying all points on a grid with resolution reso x reso according to training data
        """
        x = torch.linspace(0, 1, reso)
        y = torch.linspace(0, 1, reso)
        grid_x, grid_y = torch.meshgrid(x, y)
        grid_points = torch.stack([grid_x.flatten(), grid_y.flatten()], dim=1)
        grid_data = {"X": grid_points}
        grid_nearest_indices = self._calc_distances_to_points(
            grid_data["X"], self.hsv["X"]
        )
        grid_data["Y"] = self._classify(grid_nearest_indices, self.hsv, v)
        # print("Plotting hsv")
        fig, ax = plt.subplots(1, 1)
        ax.layout = "tight"
        self._plot(
            grid_data,
            c=["#219ebc", "#fb8500"],
            m="s",
            s=float(1000 / self.plotting_reso),
        )
        self._plot(self.hsv, c=["#8ecae6", "#ffb703"], m="o", s=50)
        ax.legend([0, 1])
        ax.set_title(f"H_sv data generating function visualized")
        try:
            os.mkdir("results")
        except Exception as e:
            pass  # print(f"Already made dir probabily, {e}")
        date_time = datetime.now().strftime("%Y%m%d_%H%M%S")
        fig.savefig(
            f"./results/{self.save_subfolder}/underlying_distribution_visualized.png"
        )
        # plt.show()
        plt.close()

    def _generate_grid(self, reso: int, v: int):
        """
        Generate regions with colors to show if test points on that area would be classified as 0 or 1
        This is done by classifiying all points on a grid with resolution reso x reso according to training data
        """
        x = torch.linspace(0, 1, reso)
        y = torch.linspace(0, 1, reso)
        grid_x, grid_y = torch.meshgrid(x, y)
        grid_points = torch.stack([grid_x.flatten(), grid_y.flatten()], dim=1)
        grid_data = {"X": grid_points}
        grid_nearest_indices = self._calc_distances_to_points(
            grid_data["X"], self.data_train["X"]
        )
        grid_data["Y"] = self._classify(grid_nearest_indices, self.data_train, v)
        return grid_data

    def _plot(self, data, c=["r", "b"], m=".", s=5):
        """
        Fast scatter plotting for tensors
        """
        X = data["X"]
        Y = data["Y"]
        # Convert Y to a NumPy array for indexing colors
        Y_np = Y.numpy() if isinstance(Y, torch.Tensor) else Y

        # Plot all points at once using scatter
        for label in set(Y_np):
            label_mask = Y_np == label  # Boolean mask for the label
            plt.scatter(
                X[label_mask, 0],
                X[label_mask, 1],
                c=c[label],
                label=label,
                marker=m,
                s=s,
            )

    def _plot_and_save(self, k, trial=0):
        """
        Plot and save the
        - H_sv centers (large points)
        - Grid (background color representing how train_data would classify new points)
        - Testing data sampled from H_sv (small points)
        """
        # print("Plotting hsv")
        fig, ax = plt.subplots(1, 1)
        ax.layout = "tight"
        self._plot(
            self.data_grid,
            c=["#219ebc", "#fb8500"],
            m="s",
            s=float(1000 / self.plotting_reso),
        )
        self._plot(self.hsv, c=["#8ecae6", "#ffb703"], m="o", s=50)
        self._plot(self.data_test, c=["#8ecae6", "#ffb703"], m=".")
        ax.legend([0, 1])

        ax.set_title(
            f"m: {self.n_train}, k: {k}, Trial: {trial} \n Acc: {self.acc:.2f}"
            + ", $n_{y=0}$: "
            + str(int((self.data_train["Y"] == 0).sum()))
            + ", $n_{y=1}$: "
            + str(int((self.data_train["Y"] == 1).sum()))
        )
        try:
            os.mkdir("results")
        except Exception as e:
            pass  # print(f"Already made dir probabily, {e}")
        date_time = datetime.now().strftime("%Y%m%d_%H%M%S")
        fig.savefig(
            f"./results/{self.save_subfolder}/{self.n_train:04d}_{k:02d}_{trial:03d}.png"
        )
        # plt.show()
        plt.close()

    def classify_and_evaluate(self, k):
        """
        Classify test data and evaluate the accuracy, plot the results if plot_flag is True
        """
        # print("evaluating")
        if self.plot_flag == True:
            self.data_grid = self._generate_grid(
                self.plotting_reso, k
            )  # grid needs to be regenerated too

        # predict error for test data
        pred_labels = self._classify(self.k_nearest_indices, self.data_train, k)
        # acc between two bool labels
        self.acc = pred_labels.eq(self.data_test["Y"]).sum() / len(pred_labels)

        if self.plot_flag:
            self._plot_and_save(k)
        return self.acc

    def _classify(self, nearest_index_matrix, ground_truth_data, v):
        """
        Classify the points based on the majority vote of the v-nearest neighbors
        """
        # labels = self.data_train['Y'][nearest_index_matrix][:,:v]  # Shape (n, v)
        labels = ground_truth_data["Y"][nearest_index_matrix][:, :v]  # Shape (n, v)
        votes = torch.sum(labels, dim=1)
        majority_votes = votes > v // 2
        return majority_votes

    def _calc_distances_to_points(
        self, all: torch.tensor, ground_truth: torch.tensor
    ) -> torch.tensor:
        """
        Calculate the distance between point and all points in all and sort them
        max_k returned

        Output:
        Matrix of shape (n, max_k) containing the indices of the k-nearest neighbors for n points
        """
        # Using broadcasting: ||A[i] - A[j]||^2 = sum((A[i] - A[j])^2) for all i, j
        distances = torch.cdist(
            all, ground_truth
        )  # Compute pairwise Euclidean distances, shape (n, n)

        sorted_distances, indices = torch.sort(distances, dim=1)

        k_nearest_indices = indices[
            :, : self.k_max
        ]  # Exclude the first column, which corresponds to the point itself
        return k_nearest_indices


# Testing knn-torch
torch.device("cuda" if torch.cuda.is_available() else "cpu")
knn = KNN(noisy=True, n_train=10000, n_test=2000, plotting_reso=200, plot_flag=False)
knn._plot_generate_grid_hsv(knn.plotting_reso, 3)
time = datetime.now()
for i in range(10, 220, 10):
    acc = knn.classify_and_evaluate(k=i)
    print(f"for k = {i}, acc: {acc}")
t_torch = datetime.now() - time
print("\nTime taken to run 20 trials with knn-torch: ", t_torch)

# Testing sk-learn's knn
from sklearn.neighbors import KNeighborsClassifier

time = datetime.now()
for i in range(10, 220, 10):
    neigh = KNeighborsClassifier(n_neighbors=i)
    neigh.fit(knn.data_train["X"], knn.data_train["Y"])
    preds = neigh.predict(knn.data_test["X"])
    acc = (preds == knn.data_test["Y"]).sum() / len(preds)
    print(f"for k = {i}, acc: {acc}")
t_sklearn = datetime.now() - time
print("\nTime taken to run 20 trials with sklearn: ", t_sklearn)

print("Speedup: ", t_sklearn / t_torch, " times")

# Rerun with plotting for visualization cases
torch.device("cuda" if torch.cuda.is_available() else "cpu")
knn = KNN(noisy=True, n_train=10000, n_test=2000, plotting_reso=200, plot_flag=True)
knn._plot_generate_grid_hsv(knn.plotting_reso, 3)
acc = knn.classify_and_evaluate(k=10)
print(f"for k = {i}, acc: {acc}")
