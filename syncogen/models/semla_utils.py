import torch


def _pad_edges(edges, max_edges, value=0):
    """Add fake edges to an edge tensor so that the shape matches max_edges

    Args:
        edges (torch.Tensor): Unbatched edge tensor, shape [num_edges, 2], each element is a node index for the edge
        max_edges (int): The number of edges the output tensor should have
        value (int): Padding value, default 0

    Returns:
        (torch.Tensor, torch.Tensor): Tuple of padded edge tensor and padding mask. Shapes [max_edges, 2] for edge
                tensor and [max_edges] for mask. Mask is one for pad elements, 0 otherwise.
    """

    num_edges = edges.size(0)
    mask_kwargs = {"dtype": torch.int64, "device": edges.device}

    if num_edges > max_edges:
        raise ValueError(
            "Number of edges in edge tensor to be padded cannot be greater than max_edges."
        )

    add_edges = max_edges - num_edges

    if add_edges == 0:
        pad_mask = torch.zeros(num_edges, **mask_kwargs)
        return edges, pad_mask

    pad = (0, 0, 0, add_edges)
    padded = torch.nn.functional.pad(edges, pad, mode="constant", value=value)

    zeros_mask = torch.zeros(num_edges, **mask_kwargs)
    ones_mask = torch.ones(add_edges, **mask_kwargs)
    pad_mask = torch.cat((zeros_mask, ones_mask), dim=0)

    return padded, pad_mask


def edges_from_adj(adj_matrix):
    """Flatten an adjacency matrix into a 1D edge representation

    Args:
        adj_matrix (torch.Tensor): Batched adjacency matrix, shape [batch_size, num_nodes, num_nodes]. It can contain
                any non-zero integer for connected nodes but must be 0 for unconnected nodes.

    Returns:
        A tuple of the edge tensor and the edge mask tensor. The edge tensor has shape [batch_size, max_num_edges, 2]
        and the mask [batch_size, max_num_edges]. The mask contains 1 for real edges, 0 otherwise.
    """

    adj_ones = torch.zeros_like(adj_matrix).int()
    adj_ones[adj_matrix != 0] = 1

    # Pad each batch element by a seperate amount so that they can all be packed into a tensor
    # It might be possible to do this in batch form without iterating, but for now this will do
    num_edges = adj_ones.sum(dim=(1, 2)).tolist()
    edge_tuples = list(adj_matrix.nonzero()[:, 1:].split(num_edges))
    padded = [_pad_edges(edges, max(num_edges), value=0) for edges in edge_tuples]

    # Unravel the padded tuples and stack them into batches
    edge_tuples_padded, pad_masks = tuple(zip(*padded))
    edges = torch.stack(edge_tuples_padded).long()
    edges = (edges[:, :, 0], edges[:, :, 1])
    edge_mask = (torch.stack(pad_masks) == 0).long()
    return edges, edge_mask


def calc_distances(coords, edges=None, sqrd=False, eps=1e-6):
    """Computes distances between connected nodes

    Takes an optional edges argument. If edges is None this will calculate distances between all nodes and return the
    distances in a batched square matrix [batch_size, num_nodes, num_nodes]. If edges is provided the distances are
    returned for each edge in a batched 1D format [batch_size, num_edges].

    Args:
        coords (torch.Tensor): Coordinate tensor, shape [batch_size, num_nodes, 3]
        edges (tuple): Two-tuple of connected node indices, each tensor has shape [batch_size, num_edges]
        sqrd (bool): Whether to return the squared distances
        eps (float): Epsilon to add before taking the square root for numical stability in the gradients

    Returns:
        torch.Tensor: Distances tensor, the shape depends on whether edges is provided (see above).
    """

    # TODO add checks

    # Create fake batch dim if unbatched
    unbatched = False
    if len(coords.size()) == 2:
        coords = coords.unsqueeze(0)
        unbatched = True

    if edges is None:
        coord_diffs = coords.unsqueeze(-2) - coords.unsqueeze(-3)
        sqrd_dists = torch.sum(coord_diffs * coord_diffs, dim=-1)

    else:
        edge_is, edge_js = edges
        batch_index = torch.arange(coords.size(0)).unsqueeze(1)
        coord_diffs = coords[batch_index, edge_js, :] - coords[batch_index, edge_is, :]
        sqrd_dists = torch.sum(coord_diffs * coord_diffs, dim=2)

    sqrd_dists = sqrd_dists.squeeze(0) if unbatched else sqrd_dists

    if sqrd:
        return sqrd_dists

    return torch.sqrt(sqrd_dists + eps)


def adj_from_node_mask(node_mask, self_connect=False):
    """Creates an edge mask from a given node mask assuming all nodes are fully connected excluding self-connections

    Args:
        node_mask (torch.Tensor): Node mask tensor, shape [batch_size, num_nodes], 1 for real node 0 otherwise
        self_connect (bool): Whether to include self connections in the adjacency

    Returns:
        torch.Tensor: Adjacency tensor, shape [batch_size, num_nodes, num_nodes], 1 for real edge 0 otherwise
    """

    num_nodes = node_mask.size()[1]

    # Outer product via broadcasting
    adjacency = node_mask.unsqueeze(2) * node_mask.unsqueeze(1)

    # Set diagonal connections
    node_idxs = torch.arange(num_nodes)
    self_mask = node_mask if self_connect else torch.zeros_like(node_mask)
    adjacency[:, node_idxs, node_idxs] = self_mask

    return adjacency


def edges_from_nodes(coords, k=None, node_mask=None, edge_format="adjacency"):
    """Constuct edges from node coords

    Connects a node to its k nearest nodes. If k is None then connects each node to all its neighbours. A node is
    never connected to itself.

    Args:
        coords (torch.Tensor): Node coords, shape [batch_size, num_nodes, 3]
        k (int): Number of neighbours to connect each node to, None means connect to all nodes except itself
        node_mask (torch.Tensor): Node mask, shape [batch_size, num_nodes], 1 for real nodes 0 otherwise
        edge_format (str): Edge format, should be either 'adjacency' or 'list'

    Returns:
        If format is 'adjacency' this returns an adjacency matrix, shape [batch_size, num_nodes, num_nodes] which
        contains 1 for connected nodes and 0 otherwise. Note that if a value for k is provided the adjacency matrix
        may not be symmetric and should always be used s.t. 'from nodes' are in dim 1 and 'to nodes' are in dim 2.

        If format is 'list' this returns the tuple (edges, edge mask), edges is also a two-tuple of tensors, each of
        shape [batch_size, num_edges], specifying node indices for each edge. The edge mask has shape
        [batch_size, num_edges] and contains 1 for 'real' edges and 0 otherwise.
    """

    if edge_format not in ["adjacency", "list"]:
        raise ValueError(f"Unrecognised edge format '{edge_format}'")

    adj_format = edge_format == "adjacency"
    batch_size, num_nodes, _ = coords.size()

    # If node mask is None all nodes are real
    if node_mask is None:
        node_mask = torch.ones(
            (batch_size, num_nodes), device=coords.device, dtype=torch.int64
        )

    adj_matrix = adj_from_node_mask(node_mask)

    if k is not None:
        # Find k closest nodes for each node
        dists = calc_distances(coords)
        dists[adj_matrix == 0] = float("inf")
        _, best_idxs = dists.topk(k, dim=2, largest=False)

        # Adjust adj matrix to only have k connections per node
        k_adj_matrix = torch.zeros_like(adj_matrix)
        batch_idxs = torch.arange(batch_size).view(-1, 1, 1).expand(-1, num_nodes, k)
        node_idxs = torch.arange(num_nodes).view(1, -1, 1).expand(batch_size, -1, k)
        k_adj_matrix[batch_idxs, node_idxs, best_idxs] = 1

        # Ensure that there are no connections to fake nodes
        k_adj_matrix[adj_matrix == 0] = 0
        adj_matrix = k_adj_matrix

    if adj_format:
        return adj_matrix

    edges, edge_mask = edges_from_adj(adj_matrix)
    return edges, edge_mask
