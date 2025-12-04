from dataclasses import dataclass
from git import Optional
import numpy as np
import torch
from torch import Tensor
from torch.distributions.laplace import Laplace
from scipy.optimize import linear_sum_assignment
from sklearn.neighbors import NearestNeighbors
from scipy.stats import wasserstein_distance_nd

import plotly.graph_objects as go

from nerfstudio.utils.poses import to4x4

EPS = 1e-6
MIN_VAR = 1e-3
MAX_COST = 1e9

@dataclass
class MultiBernoulli:

    def __init__(self, prediction) -> None:
        self.n_mb = prediction.shape[-2]
        self.existence_probabilities = prediction[..., 0].clamp(min=EPS, max=1-EPS)
        self.x = prediction[..., 1]
        self.y = prediction[..., 2]
        self.z = prediction[..., 3]
        self.x_var = prediction[..., 4].clamp(min=MIN_VAR)
        self.y_var = prediction[..., 5].clamp(min=MIN_VAR)
        self.z_var = prediction[..., 6].clamp(min=MIN_VAR)
        self.distribution = Laplace

    def get_means(self, index):
        x = self.x[..., index].squeeze(0)
        y = self.y[..., index].squeeze(0)
        z = self.z[..., index].squeeze(0)
        return torch.stack([x, y, z], dim=-1)


def calculate_radar_loss(batch: Tensor, prediction: Tensor, indices: Tensor, loss_type: str = "nll", training: bool = True):
    """Calculate the radar loss."""
    # Get indices of the ground truth radar point cloud for each scan
    seg_indices = (indices[:,1] == 0).nonzero(as_tuple=True)[0]
    num_scans = seg_indices.shape[0]
    seg_indices = torch.cat((seg_indices, torch.tensor([indices.shape[0]], device=seg_indices.device)))

    losses = []
    associations, mbs = [], []
    for i in range(num_scans):
        # Get the ground truth point cloud
        gt = batch[seg_indices[i]:seg_indices[i+1], :]
        gt = gt[..., :3]

        pred = prediction[i]
        mb = MultiBernoulli(prediction=pred)

        # Calculate the cost matrix and perform association
        if training:
            c = get_cost_matrix(gt, mb, "euclidean")
        else:
            c = get_cost_matrix(gt, mb, loss_type)
        row_ind, col_ind = linear_sum_assignment(c.cpu().detach().numpy())
        association = -torch.ones((mb.n_mb, 2), device=gt.device)
        association[:, 0] = torch.arange(mb.n_mb)
        association[torch.tensor(row_ind, dtype=torch.long, device=gt.device), 1] = torch.tensor(col_ind, device=gt.device).float()

        # Get loss for the current radar scan
        loss = get_radar_loss(gt, mb, association, loss_type)

        losses.append(loss)
        associations.append(association)
        mbs.append(mb)
    losses_tensor = torch.stack(losses)

    return torch.mean(losses_tensor), associations[-1], mbs[-1]


def get_cost_matrix(batch: Tensor, mb: MultiBernoulli, method: str = "nll") -> Tensor:
    n_target = batch.shape[0]

    if method == "euclidean":
        cart_params = mb.get_means(torch.arange(mb.n_mb, dtype=torch.int, device=batch.device))
        costs = torch.cdist(cart_params, batch[:, :3])
        log_r_k = mb.existence_probabilities.log().repeat(n_target, 1).T
        cost_matrix = costs - log_r_k

    elif method == "nll":
        log_r_k_loss = mb.existence_probabilities.log().repeat(n_target, 1).T
        log_1_r_k_loss = (1 - mb.existence_probabilities).log().repeat(n_target, 1).T
        cost_matrix = log_1_r_k_loss - log_r_k_loss

        x_pdf = mb.distribution(loc=mb.x, scale=mb.x_var)
        y_pdf = mb.distribution(loc=mb.y, scale=mb.y_var)
        z_pdf = mb.distribution(loc=mb.z, scale=mb.z_var)

        x_log_probs = x_pdf.log_prob(batch[:, 0].unsqueeze(-1)).squeeze(0).T
        y_log_probs = y_pdf.log_prob(batch[:, 1].unsqueeze(-1)).squeeze(0).T
        z_log_probs = z_pdf.log_prob(batch[:, 2].unsqueeze(-1)).squeeze(0).T

        cost_matrix += - x_log_probs - y_log_probs - z_log_probs

    if cost_matrix.isinf().sum() > 0:
        print("Inf values in cost matrix. Setting to MAX_COST.")
        cost_matrix[cost_matrix.isinf()] = MAX_COST

    return cost_matrix


def get_radar_loss(batch, mb, association, loss_type):

    associated = association[:, 1] > -1

    if loss_type == "nll":
        log_r_k_loss = mb.existence_probabilities.log().flatten()
        log_1_r_k_loss = (1 - mb.existence_probabilities).log().flatten()
        losses = -log_1_r_k_loss

        if associated.sum() > 0:

            x_pdf = mb.distribution(loc=mb.x, scale=mb.x_var)
            y_pdf = mb.distribution(loc=mb.y, scale=mb.y_var)
            z_pdf = mb.distribution(loc=mb.z, scale=mb.z_var)

            x_log_probs = x_pdf.log_prob(batch[:, 0].unsqueeze(-1)).squeeze(0).T
            y_log_probs = y_pdf.log_prob(batch[:, 1].unsqueeze(-1)).squeeze(0).T
            z_log_probs = z_pdf.log_prob(batch[:, 2].unsqueeze(-1)).squeeze(0).T

            associated_costs = - x_log_probs[associated, ...] - y_log_probs[associated, ...] - z_log_probs[associated, ...]

            if associated_costs.shape == (1,):
                associated_costs = associated_costs.view(-1, 1)

            losses[associated] = associated_costs[torch.arange(associated_costs.shape[0]), association[associated, 1].long()] - log_r_k_loss[associated]

    elif loss_type == "euclidean":

        cart_params = mb.get_means(torch.arange(mb.n_mb, dtype=torch.int, device=batch.device))
        losses = - (1 - mb.existence_probabilities).log().flatten()

        if associated.sum() > 0:
            losses[associated] = - mb.existence_probabilities.log().flatten()[associated]
            losses[associated] += torch.norm(cart_params[association[associated, 0].long(), :] - batch[association[associated, 1].long(), :3], dim=-1).flatten()

    loss = (torch.sum(losses)) / mb.n_mb
    return loss

def sample_radar_points(radar_output: Tensor, loss_type: str, threshold: Optional[float] = 0.5, max_detections: int = 1000):
    mb = MultiBernoulli(prediction=radar_output[-1, ...])

    existence_probabilities = mb.existence_probabilities.flatten()
    sorted_ep = torch.argsort(existence_probabilities, descending=True)
    if sorted_ep.shape[0] > max_detections:
        sorted_ep = sorted_ep[:max_detections]
    existence_probabilities_sorted = existence_probabilities[sorted_ep]

    if loss_type == "nll":
        ber_indices = torch.empty(0, device=mb.existence_probabilities.device).long()
        x = torch.empty((0, 1), device=mb.existence_probabilities.device)
        y = torch.empty((0, 1), device=mb.existence_probabilities.device)
        z = torch.empty((0, 1), device=mb.existence_probabilities.device)

        for i in range(mb.n_mb):
            random_prob = torch.rand(1, device=mb.existence_probabilities.device)
            if random_prob < mb.existence_probabilities[..., i] and i in sorted_ep:
                ber_indices = torch.cat([ber_indices, torch.tensor([i], device=mb.existence_probabilities.device).long()])
                x = torch.cat([x, mb.distribution(loc=mb.x[..., i], scale=mb.x_var[..., i]).rsample(torch.Size([1])).view(-1, 1)])
                y = torch.cat([y, mb.distribution(loc=mb.y[..., i], scale=mb.y_var[..., i]).rsample(torch.Size([1])).view(-1, 1)])

                if mb.z_var is not None:
                    z = torch.cat([z, mb.distribution(loc=mb.z[..., i], scale=mb.z_var[..., i]).rsample(torch.Size([1])).view(-1, 1)])
                else:
                    z = torch.cat([z, torch.zeros((1, 1), device=mb.existence_probabilities.device)])

        radar_points = torch.cat([x, y, z], dim=-1)

    elif loss_type == "euclidean":
        ber_indices = torch.arange(mb.n_mb, device=mb.existence_probabilities.device).long()
        x = mb.x.view(-1, 1)
        y = mb.y.view(-1, 1)
        if mb.z_var is not None:
            z = mb.z.view(-1, 1)
        else:
            z = torch.zeros_like(x)

        radar_points = torch.cat([x, y, z], dim=-1).view(-1, 3)[sorted_ep, :]
        ber_indices = ber_indices[sorted_ep]

        radar_points = radar_points[existence_probabilities_sorted > threshold, :]
        ber_indices = ber_indices[existence_probabilities_sorted > threshold]

    return radar_points, ber_indices

def plot_radar_samples(batch, radar_points, association, mb, ber_indices):

    fig = go.Figure()

    radar_points = radar_points.cpu().detach().numpy()
    if batch is not None:
        batch = batch[:, :3].cpu().detach().numpy()
        fig.add_trace(go.Scatter3d(x=batch[..., 0].squeeze().tolist(), y=batch[..., 1].squeeze().tolist(), z=batch[..., 2].squeeze().tolist(), mode='markers', marker=dict(color='green', size=[10], sizemode='diameter'), name='Batch'))


    if association is not None:
        association = association.cpu().detach().numpy()
        existence_probabilities = mb.existence_probabilities.cpu().detach().numpy()
        ber_indices = ber_indices.cpu().detach().numpy()
        association = association[ber_indices, :]

        for b in range(radar_points.shape[0]):
            index = int(association[b, 0])
            if association[b, 1] > -1:
                fig.add_trace(go.Scatter3d(x=[radar_points[b, 0]], y=[radar_points[b, 1]], z=[radar_points[b, 2]], opacity=(existence_probabilities[index]).item(), mode='markers', marker=dict(color='blue', size=[10], sizemode='diameter'), name=f'Radar Point {index}'))
            else:
                fig.add_trace(go.Scatter3d(x=[radar_points[b, 0].tolist()], y=[radar_points[b, 1].tolist()], z=[radar_points[b, 2].tolist()], opacity=(existence_probabilities[index]).item(), mode='markers', marker=dict(color='red', size=[10], sizemode='diameter'), name=f'Radar Point {index}'))

    else:
        fig.add_trace(go.Scatter3d(x=radar_points[..., 0].squeeze(), y=radar_points[..., 1].squeeze(), z=radar_points[..., 2].squeeze(), mode='markers', marker=dict(color='blue', size=[10], sizemode='diameter'), name='Radar points'))


    fig.update_layout(
        xaxis_title="x [m]",
        yaxis_title="y [m]",
        xaxis_range=[-10,100],
        yaxis_range=[-50,50],
        plot_bgcolor='white',
        hovermode='closest',
        hoverlabel=dict(bgcolor="white", font_size=12),
        margin=dict(l=50, r=50, t=50, b=50),
        template='plotly_white',
        scene=dict(aspectmode="data")
    )

    return fig

def add_actor_boxes_to_figure(figure, actor_information, r2w, radar_idx):
    actor_b2w = actor_information["actor_b2w"]
    actor_indices = actor_information["actor_indices"]
    actor_sizes = actor_information["actor_sizes"]
    scan_indices = actor_information["scan_indices"]

    actor_b2w = actor_b2w[scan_indices == radar_idx]
    actor_indices = actor_indices[scan_indices == radar_idx]

    for i in range(len(actor_indices)):
        actor_size = actor_sizes[actor_indices[i]]
        actor_b2w_i = actor_b2w[i]
        actor_box = torch.tensor(
            [
                [-actor_size[0] / 2, -actor_size[1] / 2, -actor_size[2] / 2],
                [-actor_size[0] / 2, -actor_size[1] / 2, actor_size[2] / 2],
                [-actor_size[0] / 2, actor_size[1] / 2, -actor_size[2] / 2],
                [-actor_size[0] / 2, actor_size[1] / 2, actor_size[2] / 2],
                [actor_size[0] / 2, -actor_size[1] / 2, -actor_size[2] / 2],
                [actor_size[0] / 2, -actor_size[1] / 2, actor_size[2] / 2],
                [actor_size[0] / 2, actor_size[1] / 2, -actor_size[2] / 2],
                [actor_size[0] / 2, actor_size[1] / 2, actor_size[2] / 2],
            ],
            device=actor_b2w_i.device,
        )
        actor_box = torch.cat((actor_box, torch.ones_like(actor_box[:, :1], device=actor_b2w_i.device)), dim=1)
        actor_box = torch.linalg.inv(to4x4(r2w.to(actor_b2w_i.device))) @ actor_b2w_i @ actor_box.t()
        actor_box = actor_box.t()
        actor_box = actor_box.detach().cpu().numpy()

        edges = [
            (0, 1), (1, 3), (3, 2), (2, 0),  # Bottom face
            (4, 5), (5, 7), (7, 6), (6, 4),  # Top face
            (0, 4), (1, 5), (2, 6), (3, 7)   # Side edges
        ]

        x_lines, y_lines, z_lines = [], [], []

        for edge in edges:
            for idx in edge:
                x_lines.append(actor_box[idx, 0])
                y_lines.append(actor_box[idx, 1])
                z_lines.append(actor_box[idx, 2])
            # None to break the line and start a new one
            x_lines.append(None)
            y_lines.append(None)
            z_lines.append(None)

        figure.add_trace(go.Scatter3d(
            x=x_lines,
            y=y_lines,
            z=z_lines,
            mode="lines",
            line=dict(color="Black", width=4),
        ))

    return figure

def chamfer_distance(x, y, metric='l2', direction='bi'):
    """Chamfer distance between two point clouds
    source: https://gist.github.com/sergeyprokudin/c4bf4059230da8db8256e36524993367
    Parameters
    ----------
    x: numpy array [n_points_x, n_dims]
        first point cloud
    y: numpy array [n_points_y, n_dims]
        second point cloud
    metric: string or callable, default l2
        metric to use for distance computation. Any metric from scikit-learn or scipy.spatial.distance can be used.
    direction: str
        direction of Chamfer distance.
            'y_to_x':  computes average minimal distance from every point in y to x
            'x_to_y':  computes average minimal distance from every point in x to y
            'bi': compute both
    Returns
    -------
    chamfer_dist: float
        computed bidirectional Chamfer distance:
            sum_{x_i in x}{min_{y_j in y}{||x_i-y_j||**2}} + sum_{y_j in y}{min_{x_i in x}{||x_i-y_j||**2}}
    """

    if direction=='y_to_x':
        x_nn = NearestNeighbors(n_neighbors=1, leaf_size=1, algorithm='kd_tree', metric=metric).fit(x)
        min_y_to_x = x_nn.kneighbors(y)[0]
        chamfer_dist = np.mean(min_y_to_x)
    elif direction=='x_to_y':
        y_nn = NearestNeighbors(n_neighbors=1, leaf_size=1, algorithm='kd_tree', metric=metric).fit(y)
        min_x_to_y = y_nn.kneighbors(x)[0]
        chamfer_dist = np.mean(min_x_to_y)
    elif direction=='bi':
        x_nn = NearestNeighbors(n_neighbors=1, leaf_size=1, algorithm='kd_tree', metric=metric).fit(x)
        min_y_to_x = x_nn.kneighbors(y)[0]
        y_nn = NearestNeighbors(n_neighbors=1, leaf_size=1, algorithm='kd_tree', metric=metric).fit(y)
        min_x_to_y = y_nn.kneighbors(x)[0]
        chamfer_dist = np.mean(min_y_to_x) + np.mean(min_x_to_y)
    else:
        raise ValueError("Invalid direction type. Supported types: \'y_x\', \'x_y\', \'bi\'")

    return chamfer_dist

def emd_distance(x, y):
    # Compute Wasserstein distance
    distance = wasserstein_distance_nd(x, y)
    return distance