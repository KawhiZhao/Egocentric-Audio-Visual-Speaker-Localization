import numpy as np
from scipy.spatial import distance
from scipy.optimize import linear_sum_assignment
import pdb
import torch
import torch.nn as nn
import torch.nn.functional as F

def modified_nms(points, threshold_distance, peak_confidence):
    """
    Perform modified non-maximum suppression (NMS) on the points based on confidence and distance,
    retaining only the valid points where [0][i][j] < [1][i][j].

    Args:
        points (np.ndarray): 3D numpy array of points where [0][i][j] represents confidence and
                             [1][i][j] represents confidence as well.
        threshold_distance (float): Distance threshold value.

    Returns:
        List[Tuple[int, int]]: List of filtered point coordinates after modified NMS.
    """
    selected_points = []

    _, height, width = points.shape
    # pdb.set_trace()
    for i in range(height):
        for j in range(width):
            if points[0][i][j] < points[1][i][j] and points[1][i][j] >= peak_confidence:
                if is_local_maximum(points, i, j, threshold_distance):
                    selected_points.append([i, j])

    return selected_points

def is_local_maximum(points, i, j, threshold_distance):
    """
    Check if the point at position (i, j) is a local maximum within the specified distance threshold.

    Args:
        points (np.ndarray): 3D numpy array of points where [0][i][j] represents confidence and
                             [1][i][j] represents confidence as well.
        i (int): Row index of the point.
        j (int): Column index of the point.
        threshold_distance (float): Distance threshold value.

    Returns:
        bool: True if the point is a local maximum, False otherwise.
    """
    confidence = points[1][i][j]

    for x in range(i - threshold_distance, i + threshold_distance + 1):
        for y in range(j - threshold_distance, j + threshold_distance + 1):
            if 0 <= x < points.shape[1] and 0 <= y < points.shape[2]:
                if confidence < points[1][x][y] and points[0][x][y] < points[1][x][y]:
                    return False

    return True

def modified_nms_v2(points, threshold_distance, peak_confidence):
    """
    Perform modified non-maximum suppression (NMS) on the points based on confidence and distance,
    retaining only the valid points where [0][i][j] < [1][i][j].

    Args:
        points (np.ndarray): 3D numpy array of points where [0][i][j] represents confidence and
                             [1][i][j] represents confidence as well.
        threshold_distance (float): Distance threshold value.

    Returns:
        List[Tuple[int, int]]: List of filtered point coordinates after modified NMS.
    """
    selected_points = []

    _, height, width = points.shape
    # pdb.set_trace()
    points = points.squeeze(0)
    for i in range(height):
        for j in range(width):
            if points[i][j] >= peak_confidence:
                if is_local_maximum_v2(points, i, j, threshold_distance):
                    selected_points.append([i, j])

    return selected_points

def is_local_maximum_v2(points, i, j, threshold_distance):
    """
    Check if the point at position (i, j) is a local maximum within the specified distance threshold.

    Args:
        points (np.ndarray): 3D numpy array of points where [0][i][j] represents confidence and
                             [1][i][j] represents confidence as well.
        i (int): Row index of the point.
        j (int): Column index of the point.
        threshold_distance (float): Distance threshold value.

    Returns:
        bool: True if the point is a local maximum, False otherwise.
    """
    confidence = points[i][j]

    for x in range(i - threshold_distance, i + threshold_distance + 1):
        for y in range(j - threshold_distance, j + threshold_distance + 1):
            if 0 <= x < points.shape[0] and 0 <= y < points.shape[1]:
                if confidence < points[x][y]:
                    return False

    return True

def compute_distances(A, B):
    """
    Compute the minimum distances between sets A and B.

    Args:
        A (list): List of points in set A, where each point is represented as (x, y).
        B (list): List of points in set B, where each point is represented as (x, y).

    Returns:
        float: Minimum distance from A to B.
        float: Mean of the minimum distances from A to B.
        float: Standard deviation of the minimum distances from A to B.
        float: Minimum distance from B to A.
        float: Mean of the minimum distances from B to A.
        float: Standard deviation of the minimum distances from B to A.
    """
    if len(A) == 0 or len(B) == 0:
        # Handle the case where either A or B is an empty set
        return (np.nan, np.nan, np.nan, np.nan, np.nan, np.nan)

    # Convert A and B to numpy arrays for efficient computation
    A = np.array(A)
    B = np.array(B)

    # Compute pairwise Euclidean distances
    pairwise_distances = distance.cdist(A, B, metric='euclidean')

    # Compute minimum distances from A to B
    min_distances_A_to_B = np.min(pairwise_distances, axis=1)
    min_distance_A_to_B = np.min(min_distances_A_to_B)
    mean_distance_A_to_B = np.mean(min_distances_A_to_B)
    std_distance_A_to_B = np.std(min_distances_A_to_B)
    # pdb.set_trace()
    # Compute minimum distances from B to A
    min_distances_B_to_A = np.min(pairwise_distances, axis=0)
    min_distance_B_to_A = np.min(min_distances_B_to_A)
    mean_distance_B_to_A = np.mean(min_distances_B_to_A)
    std_distance_B_to_A = np.std(min_distances_B_to_A)

    return (min_distance_A_to_B, mean_distance_A_to_B, std_distance_A_to_B,
            min_distance_B_to_A, mean_distance_B_to_A, std_distance_B_to_A)

class ScaledDotProductAttention(nn.Module):
    ''' Scaled Dot-Product Attention '''

    def __init__(self, temperature, attn_dropout=0.1):
        super().__init__()
        self.temperature = temperature
        self.dropout = nn.Dropout(attn_dropout)

    def forward(self, q, k, v, mask=None):

        attn = torch.matmul(q / self.temperature, k.transpose(2, 3))

        if mask is not None:
            attn = attn.masked_fill(mask == 0, -1e9)

        attn = self.dropout(F.softmax(attn, dim=-1))
        output = torch.matmul(attn, v)

        return output, attn

class MultiHeadAttention(nn.Module):
    ''' Multi-Head Attention module '''

    def __init__(self, n_head, d_model, d_k, d_v, dropout=0.1):
        super().__init__()

        self.n_head = n_head
        self.d_k = d_k
        self.d_v = d_v

        self.w_qs = nn.Linear(d_model, n_head * d_k, bias=False)
        self.w_ks = nn.Linear(d_model, n_head * d_k, bias=False)
        self.w_vs = nn.Linear(d_model, n_head * d_v, bias=False)
        self.fc = nn.Linear(n_head * d_v, d_model, bias=False)

        self.attention = ScaledDotProductAttention(temperature=d_k ** 0.5)

        self.dropout = nn.Dropout(dropout)
        self.layer_norm = nn.LayerNorm(d_model, eps=1e-6)


    def forward(self, q, k, v, mask=None):

        d_k, d_v, n_head = self.d_k, self.d_v, self.n_head
        sz_b, len_q, len_k, len_v = q.size(0), q.size(1), k.size(1), v.size(1)

        residual = q

        # Pass through the pre-attention projection: b x lq x (n*dv)
        # Separate different heads: b x lq x n x dv
        q = self.w_qs(q).view(sz_b, len_q, n_head, d_k)
        k = self.w_ks(k).view(sz_b, len_k, n_head, d_k)
        v = self.w_vs(v).view(sz_b, len_v, n_head, d_v)

        # Transpose for attention dot product: b x n x lq x dv
        q, k, v = q.transpose(1, 2), k.transpose(1, 2), v.transpose(1, 2)

        if mask is not None:
            mask = mask.unsqueeze(1)   # For head axis broadcasting.

        q, attn = self.attention(q, k, v, mask=mask)

        # Transpose to move the head dimension back: b x lq x n x dv
        # Combine the last two dimensions to concatenate all the heads together: b x lq x (n*dv)
        q = q.transpose(1, 2).contiguous().view(sz_b, len_q, -1)
        q = self.dropout(self.fc(q))
        q += residual

        q = self.layer_norm(q)

        return q
