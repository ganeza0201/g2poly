import networkx as nx
import numpy as np
import numpy.linalg as lg
import pulp
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from math import *


# Golden ratio.
phi = (1 + sqrt(5)) / 2


def is_triconnected(
    graph
):
    """
    Determines if a given undirected simple graph is triconnected or not.

    Parameters
    ----------
    graph: nx.Graph

    Returns
    -------
    : bool
        Whether the given graph is triconnected or not.
    """
    if not isinstance(graph, nx.Graph):
        print("'graph' must be an undirected simple graph.")
    node_list = list(graph.nodes)
    for i in range(graph.number_of_nodes()):
        u = node_list[i]
        for j in range(i + 1, graph.number_of_nodes()):
            v = node_list[j]
            # print(u, end=", ")
            # print(v, end=": ")
            H = nx.restricted_view(graph, nodes=[u, v], edges=[])
            if not nx.algorithms.is_connected(H):
                # print("Separated !")
                return False
            else:
                # print("Still Connected !")
                pass
    return True


def get_faces(
    planar
):
    """
    Returns faces of a planar embedding.

    Parameters
    ----------
    planar: nx.PlanarEmbedding

    Returns
    -------
    faces: dict
        A dict whose values are lists representing facial walks of
        the planar embedding and whose keys are its dual nodes.
    """
    if not isinstance(planar, nx.PlanarEmbedding):
        print("'planar' must be a planar embedding.")

    # Traverse all faces of 'planar'.
    half_edge_list = list(planar.edges)
    faces = {}
    while len(half_edge_list) > 0:
        u, v = half_edge_list.pop(0)
        planar.edges[u, v]['dual_node'] = len(faces.keys())
        face = planar.traverse_face(u, v)
        for i in range(1, len(face)):
            u = face[i]
            v = face[(i + 1) % len(face)]
            planar.edges[u, v]['dual_node'] = len(faces)
            half_edge_list.remove((u, v))
        faces[len(faces.keys())] = face

    return faces


def get_dual(
    primal
):
    """
    Returns the dual of a planar embedding.

    Parameters
    ----------
    primal: nx.PlanarEmbedding

    Returns
    -------
    primal_faces: dict
        A dict whose values are lists representing facial walks of
        the primal and whose keys are its dual nodes.
    dual: nx.PlanarEmbedding
        The dual of the primal.
    dual_faces: dict
        A dict whose values are lists representing facial walks of
        the dual and whose keys are nodes of the primal.
    """
    if not isinstance(primal, nx.PlanarEmbedding):
        print("'primal' must be a planar embedding.")

    # Constracting the dual embedding of 'primal'.
    primal_faces = get_faces(primal)
    dual = nx.PlanarEmbedding()
    for tail, primal_face in primal_faces.items():
        # Define a clockwize rotation around tail in dual nodes.
        pre_head = None
        for i in range(len(primal_face)):
            u = primal_face[i]
            v = primal_face[(i + 1) % len(primal_face)]
            head = primal.edges[v, u]['dual_node']
            dual.add_half_edge_cw(tail, head, pre_head)
            pre_head = head
            primal.edges[v, u]['right_half_edge'] = (tail, head)
            dual.edges[tail, head]['dual_node'] = u
            dual.edges[tail, head]['right_half_edge'] = (u, v)

    # A face of 'dual' can be derived from
    # clockwize rotation around a node of 'primal'.
    dual.check_structure()
    dual_faces = {}
    for u in primal.nodes:
        dual_face = []
        for v in primal.neighbors_cw_order(u):
            dual_face.append(primal.edges[u, v]['right_half_edge'][0])
        dual_faces[u] = dual_face

    return primal_faces, dual, dual_faces


def get_radial(
    primal
):
    """
    Returns the dual and the radial of a planar embedding.
    The radial graph of a planar embedding G is also known as the vertex-face map
    whose vertices are the vertices of G together with the faces of G,
    and whose edges correspond to the vertex-face incidence in G.

    Parameters
    ----------
    primal: nx.PlanarEmbedding

    Returns
    -------
    dual: nx.PlanarEmbedding
        The dual of the primal.
    radial: nx.PlanarEmbedding
        The radial of the primal.
    """
    primal_faces, dual, dual_faces = get_dual(primal)
    radial = nx.PlanarEmbedding()
    for p in primal.nodes:
        p_name = 'p' + str(p)
        pre_d_name = None
        for d in dual_faces[p]:
            d_name = 'd' + str(d)
            radial.add_half_edge_cw(p_name, d_name, pre_d_name)
            pre_d_name = d_name
        radial.nodes[p_name]['primal'] = True
        radial.nodes[p_name]['original'] = p
    for d in dual.nodes:
        d_name = 'd' + str(d)
        pre_p_name = None
        for p in primal_faces[d]:
            p_name = 'p' + str(p)
            radial.add_half_edge_cw(d_name, p_name, pre_p_name)
            pre_p_name = p_name
        radial.nodes[d_name]['primal'] = False
        radial.nodes[d_name]['original'] = d
    radial.check_structure()
    return dual, radial


def circle_pack(
    graph=None,
    primal=None,
    epsilon=0.0000001
):
    """
    Calculate the simultaneous circle packing of a triconnected planar graph or planar embedding.
    An algorithm used in this method is described in [1].

    Parameters
    ----------
    graph: nx.Graph
        A triconnected (polyhedral) planar graph to circle pack.
        Since it is triconnected, it has a unique planar embedding.
        The planar embedding will be stored as a primal planar embedding.
        Simultaneously, its dual will also be circle packed.
    primal: nx.PlanarEmbedding
        A triconnected (polyhedral) planar embedding to circle pack.
        Simultaneously, its dual will also be circle packed.
    epsilon: float
        A tolerance of the sum of squared packing errors (named 'theta')
        around nodes of the primal and its dual.

    Returns
    -------
    graph: nx.Graph
        A underlying graph of the primal planar embedding.
    primal: nx.PlanarEmbedding
        A planar embedding of a given graph.
    primal_cpack: dict
        A circle packing data dicts of the primal.
    dual: nx.PlanarEmbedding
        The dual of the primal.
    dual_cpack: dict
        A circle packing data dicts of the dual.

    References
    ----------
    ..[1] Bojan Mohar.
        CIRCLE PACKING OF MAPS - THE EUCLIDEAN CASE.
        1997
        https://www.mate.polimi.it/smf/vol67/mohar.pdf
    """
    if graph is not None:
        if primal is not None:
            raise Warning("'graph' and 'primal' are simultaneously defined.")
        else:
            if not isinstance(graph, nx.Graph):
                raise Warning("'graph' must be an undirected simple graph.")
            if not is_triconnected(graph=graph):
                raise Warning("'graph' must be triconnected.")
            is_planar, primal = nx.algorithms.check_planarity(graph, counterexample=False)
            if not is_planar:
                raise Warning("'graph' must be planar.")
    else:
        if primal is None:
            raise Warning("Neither 'graph' nor 'primal' are not defined.")
        else:
            if not isinstance(primal, nx.PlanarEmbedding):
                raise Warning("'primal' must be a planar embedding.")
            graph = primal.to_undirected()
            if not is_triconnected(graph):
                raise Warning("'primal' must be polyhedral.")

    dual, radial = get_radial(primal)

    # 'radial' is an instance of nx.PlanarEmbedding all of whose faces are quadrangular.
    # Thus, a degree 3 node always exists and choose it as a node at infinity, 'infty'.
    # First, deal with nodes of 'radial' other than 'infty'.
    cpack = {}
    infty = None
    for u in radial.nodes:
        # Note that nx.PlanarEmbedding is a subclass of nx.DiGraph.
        if infty is None and radial.out_degree[u] == 3:
            infty = u
            continue
        cpack[u] = {'radius': 1, 'bounded': True, 'outer': False}
    if infty is None:
        raise Warning("No degree 3 node !")
    outer = list(radial.neighbors_cw_order(infty))
    for v in outer:
        cpack[v]['outer'] = True

    # Calculate radii of nodes of 'radial' other than 'infty'.
    while True:
        theta_sq_sum = 0
        for u in cpack.keys():
            phi = 0
            r_u = cpack[u]['radius']
            for v in radial.neighbors_cw_order(u):
                if v != infty:
                    r_v = cpack[v]['radius']
                    phi += atan(r_v / r_u)
            if cpack[u]['outer']:
                theta = phi - pi / 6
            else:
                theta = phi - pi
            cpack[u]['theta'] = theta
            theta_sq_sum += theta ** 2
        # print("theta_sq_sum: %f" % theta_sq_sum)
        if theta_sq_sum <= epsilon:
            for u in cpack.keys():
                cpack[u].pop('outer')
                cpack[u].pop('theta')
                cpack[u].pop('temp_radius')
                cpack[u].pop('temp_theta')
            break
        else:
            cpack = dict(sorted(cpack.items(),
                                key=lambda item: item[1]['theta'],
                                reverse=True))
            sigma = -inf
            t = 0
            datas = list(cpack.values())
            for i in range(len(datas) - 1):
                theta = datas[i]['theta']
                theta_succ = datas[i + 1]['theta']
                if sigma < theta - theta_succ:
                    sigma = theta - theta_succ
                    t = i
            theta_sum = 0
            nodes = list(cpack.keys())
            for i, u in enumerate(nodes):
                if i <= t:
                    theta_sum += cpack[u]['theta']
                else:
                    break
            a_0 = 1
            a_1 = 2 * (2 + max(data['radius'] for data in datas[t + 1:]))
            while True:
                a = (a_0 + a_1) / 2
                for i, u in enumerate(nodes):
                    if i <= t:
                        cpack[u]['temp_radius'] = a * cpack[u]['radius']
                    else:
                        cpack[u]['temp_radius'] = cpack[u]['radius']
                f_a = 0
                for i, u in enumerate(nodes):
                    phi = 0
                    r_u = cpack[u]['temp_radius']
                    for v in radial.neighbors_cw_order(u):
                        if v != infty:
                            r_v = cpack[v]['temp_radius']
                            phi += atan(r_v / r_u)
                    if cpack[u]['outer']:
                        temp_theta = phi - pi / 6
                    else:
                        temp_theta = phi - pi
                    cpack[u]['temp_theta'] = temp_theta
                    if i <= t:
                        f_a += cpack[u]['theta'] - cpack[u]['temp_theta']
                suitable = True
                for u in nodes[:t + 1]:
                    for v in nodes[t + 1:]:
                        if cpack[u]['temp_theta'] < cpack[v]['temp_theta']:
                            suitable = False
                            break
                    if not suitable:
                        break
                if suitable and f_a < min(sigma, theta_sum) / 3:
                    suitable = False
                if suitable:
                    break
                else:
                    exists = False
                    for u in nodes[:t + 1]:
                        for v in nodes[t + 1:]:
                            if cpack[u]['temp_theta'] < cpack[v]['temp_theta'] + sigma / 7:
                                exists = True
                                break
                        if exists:
                            break
                    if exists:
                        a_1 = a
                    else:
                        a_0 = a
            temp_r_min = min(data['temp_radius'] for data in datas)
            for u in cpack.keys():
                cpack[u]['radius'] = cpack[u]['temp_radius'] / temp_r_min

    # 'radial' neighbors around 'infty' is a equilateral triangle.
    # Thus, we can determine radius of 'infty' as the center of its inscribed circle.
    r_infty = ((cpack[outer[0]]['radius'] + cpack[outer[1]]['radius'] + cpack[outer[2]]['radius']) / 3) / sqrt(3)
    cpack[infty] = {'radius': r_infty, 'bounded': False}

    # Calculate centers of each 'radial' nodes like a breadth-first search algorithm.
    cpack[infty]['center'] = np.array([0.0, 0.0])
    for i, v in enumerate(outer):
        r_v = cpack[v]['radius']
        dist = sqrt(r_infty ** 2 + r_v ** 2)
        cpack[v]['center'] = dist * np.array([-sin(2 * i * pi / 3), cos(2 * i * pi / 3)])
        radial.nodes[v]['first_nbr'] = infty
    queue = outer
    while len(queue) > 0:
        u = queue.pop(0)
        r_u = cpack[u]['radius']
        cnt_u = cpack[u]['center']
        first_nbr = radial.nodes[u]['first_nbr']
        if first_nbr == infty:
            vec = cpack[u]['center']
        else:
            vec = cpack[first_nbr]['center'] - cnt_u
        arg = atan2(vec[1], vec[0])
        for i, v in enumerate(list(radial.neighbors_cw_order(u))):
            r_v = cpack[v]['radius']
            if i == 0:
                # 'v' is a first neighbor of 'u'.
                if v == infty:
                    # 'infty' appears only if i == 0.
                    phi = 5 * pi / 6
                else:
                    phi = atan(r_v / r_u)
                arg -= phi
            else:
                # 'v' is not a first neighbor of 'u'.
                phi = atan(r_v / r_u)
                arg -= phi
                dist = sqrt(r_u ** 2 + r_v ** 2)
                if 'center' not in cpack[v]:
                    queue.append(v)
                    cnt_v = cnt_u + dist * np.array([cos(arg), sin(arg)])
                    cpack[v]['center'] = cnt_v
                    radial.nodes[v]['first_nbr'] = u
                arg -= phi

    nx.set_node_attributes(radial, cpack)
    primal_cpack = {}
    dual_cpack = {}
    for u in radial.nodes:
        cpack_u = {'radius': radial.nodes[u]['radius'],
                   'center': radial.nodes[u]['center'],
                   'bounded': radial.nodes[u]['bounded'],
                   'normal': None,  # will be used if transformed circle is a straight line.
                   'point': None,  # will be used if transformed circle is a straight line.
                   }
        orig = radial.nodes[u]['original']
        if radial.nodes[u]['primal']:
            primal_cpack[orig] = cpack_u
        else:
            dual_cpack[orig] = cpack_u

    return graph, primal, primal_cpack, dual, dual_cpack


def disks_to_spheres(
    disks
):
    """
    Map disks on a extended plane ℝ² ∪ {∞} to the unit sphere centered at the origin
    𝕊² in ℝ³ ∪ {∞} by the inverse of stereographic projection σ: ℝ² ∪ {∞} -> 𝕊².
    For each disk in disks, its image is expressed as the intersection of 𝕊² and
    the interior or the exterior of an extended sphere which intersects 𝕊² orthogonally.

    Parameters
    ----------
    disks: dict
        A data dict for disks on a extended plane ℝ² ∪ {∞}.

    Returns
    -------
    spheres: dict
        A data dict for spheres in a extended Euclidean space
        ℝ³ ∪ {∞} all of which intersect 𝕊² orthogonally.
    """
    spheres = {}
    for v, disk in disks.items():
        if isfinite(disk['radius']):
            # 'disk' is a disk on the plane.
            R = np.sum(np.square(disk['center'])) + 1 - (disk['radius'] ** 2)
            if R != 0.0:
                # If R is nonzero, then the image of 'disk' via σ is not a hemisphere of 𝕊².
                # 'cnt' is the center of a sphere which intersects 𝕊² orthogonally such that a disk on 𝕊²
                # pasted via σ is the intersection of the interior or the exterior of it and 𝕊² .
                cnt = (np.sign(R) / abs(R)) * np.array([-2 * disk['center'][0], -2 * disk['center'][1], R - 2])
                r = sqrt((lg.norm(cnt, ord=2) ** 2) - 1)
                # 'sgn' is a sign which determines if the disk is contained in the sphere (+) or not (-).
                if disk['bounded']:
                    if 1.0 > cnt[2]:
                        sgn = +1
                    else:
                        sgn = -1
                else:
                    if 1.0 > cnt[2]:
                        sgn = -1
                    else:
                        sgn = +1
                # The extended sphere is not a plane passing through the origin and ∞.
                nml = np.array([0., 0., 0.])
            else:
                # If R is nonzero, then the image of 'disk' via σ is a hemisphere of 𝕊².
                # If 'disk' on the plane is bounded, then the pasted disk on 𝕊² must not contain
                # the north pole [0, 0, 1] of 𝕊², which means that if we make a normal unit vector
                # of a extended plane points to the hemisphere, the its z-coordinate must be negative.
                nml = np.array([disk['center'][0], disk['center'][1], 1.0]) \
                      / sqrt((disk['center'][0] ** 2) + (disk['center'][1] ** 2) + 1)
                if disk['bounded']:
                    nml *= -1
                sgn = +1
                # The extended sphere is not a sphere in Euclidean space.
                cnt = inf * nml
                r = inf
        else:
            # If 'disk' is a half plane, then the image of its boundary becomes
            # a small circle on 𝕊² that passes through the north pole [0, 0, 1].
            # Note that the boundary (a line) passes through disk['point']
            # and its normal unit vector 'disk['normal']' points to the half region.
            if disk['point'][0] != 0.0 or disk['point'][1] != 0.0:
                # If disk['point'] is other than the origin of the plane,
                # then the boundary is not a great circle that passes through
                # both the north pole and the south pole [0, 0, -1].
                inr = np.dot(disk['normal'], disk['point'])
                sgn = np.sign(inr)
                cnt = (sgn / inr) * np.array([disk['normal'][0], disk['normal'][1], inr])
                r = sqrt((lg.norm(cnt, ord=2) ** 2) - 1)
                # The extended sphere is not a plane passing through the origin and ∞.
                nml = np.array([0., 0., 0.])
            else:
                # If disk['point'] is the origin of the plane,
                # then the boundary is a great circle that passes through
                # both the north pole and the south pole.
                nml = disk['normal'].copy()
                sgn = +1
                # The extended sphere is not a sphere in Euclidean space.
                cnt = inf * nml
                r = inf
        sphere = {'center': cnt,
                  'radius': r,
                  'normal': nml,
                  'sign': sgn}
        spheres[v] = sphere
    return spheres


def transform_spheres(
    spheres,
    viewpoint,
    _return_spheres=True,
    _calc_grad=False,
    _return_value=False
):
    """
    Transform extended spheres in ℝ³ ∪ {∞} intersecting the unit sphere 𝕊²
    centered at the origin orthogonally by a Möbius transformation on Poincaré ball model
    which preserves the straight line connecting the origin and a given viewpoint
    and maps the viewpoint to the origin.

    Parameters
    ----------
    spheres: dict
        A data dict for extended spheres in ℝ³ ∪ {∞} all of which intersect 𝕊² orthogonally.
    viewpoint: np.ndarray
        A viewpoint for a Möbius transformation which preserves the straight line
        connecting the origin and it and maps it to the origin.
    _return_spheres: bool
        If this parameter is true, then this method returns transformed spheres.
    _calc_grad: bool
        If _return_spheres and this parameter is true, then this method calculates
        a gradient vector of each transformed radius as a function of the viewpoint.
    _return_value: bool
        If this parameter is true, then this method returns the minimum transformed radius.

    Returns
    -------
    new_spheres: dict
        A data dict for transformed spheres in ℝ³ ∪ {∞} by the Möbius
        transformation all of which still intersect 𝕊² orthogonally.
    r_min: float
        The minimum radius among transformed spheres.
    """
    if _return_spheres:
        new_spheres = {}
    if _return_value:
        r_min = inf
    dist = lg.norm(viewpoint, ord=2)
    for u, sphere in spheres.items():
        if isfinite(sphere['radius']):
            inr = np.dot(viewpoint, sphere['center'])
            diff = lg.norm(viewpoint - sphere['center'], ord=2)
            if diff != sphere['radius']:
                nmr = sphere['radius'] * (1.0 - (dist ** 2))
                dnm = (diff ** 2) - (sphere['radius'] ** 2)
                r = nmr / abs(dnm)
                if _return_spheres:
                    cnt = (2 * (1 - inr) * viewpoint + ((dist ** 2) - 1) * sphere['center']) / dnm
                    sgn = np.sign(dnm) * sphere['sign']
                    nml = np.array([0., 0., 0.])
                    if _calc_grad:
                        grad = (np.sign(dnm) / (dnm ** 2)) \
                               * (-2 * sphere['radius'] * dnm * viewpoint - nmr * 2 * (viewpoint - sphere['center']))
            else:
                r = inf
                if _return_spheres:
                    nml = 2 * (1 - inr) * viewpoint + ((dist ** 2) - 1) * sphere['center']
                    nml /= sphere['sign'] * lg.norm(nml, ord=2)
                    cnt = inf * nml
                    sgn = +1
                    if _calc_grad:
                        grad = np.array([0., 0., 0.])
        else:
            inr = np.dot(viewpoint, sphere['normal'])
            if inr != 0.0:
                nrm = lg.norm(sphere['normal'], ord=2)
                nmr = nrm * (1.0 - (dist ** 2))
                dnm = 2 * inr
                r = nmr / abs(dnm)
                if _return_spheres:
                    cnt = (-2 * inr * viewpoint + ((dist ** 2) - 1) * sphere['normal'])
                    sgn = -np.sign(inr)
                    nml = np.array([0., 0., 0.])
                    if _calc_grad:
                        grad = (np.sign(dnm) / (dnm ** 2)) * (-2 * nrm * dnm * viewpoint - nmr * 2 * sphere['normal'])
            else:
                r = inf
                if _return_spheres:
                    nml = sphere['normal'].copy()
                    cnt = inf * nml
                    sgn = +1
                    if _calc_grad:
                        grad = np.array([0., 0., 0.])
        if _return_spheres:
            new_sphere = {'center': cnt,
                          'radius': r,
                          'normal': nml,
                          'sign': sgn}
            if _calc_grad:
                new_sphere['gradient'] = grad
            new_spheres[u] = new_sphere
        if _return_value and r < r_min:
            r_min = r
    if _return_spheres and _return_value:
        return new_spheres, r_min
    elif _return_spheres:
        return new_spheres
    elif _return_value:
        return r_min


def maximize_minimum_sphere(
    spheres,
    epsilon=0.000001
):
    """
    Maximize the minimum radius among extended spheres in ℝ³ ∪ {∞} intersecting the unit sphere
    𝕊² centered at the origin orthogonally by Möbius transformations on Poincaré ball model.
    An algorithm used in this method is smooth quasiconvex programming described in [1], [2].

    Parameters
    ----------
    spheres: dict
        A data dict for extended spheres in ℝ³ ∪ {∞} all of which intersect 𝕊² orthogonally.
    epsilon: float
        A tolerance used in maximizing the minimum radius among extended spheres.

    Returns
    -------
    new_spheres: dict
        A data dict for transformed spheres in ℝ³ ∪ {∞} by an optimal Möbius
        transformation which maximizes the minimum radius among them.
    viewpoint: np.ndarray
        A viewpoint for the optimal Möbius transformation.

    References
    ----------
    ..[1] Marshall Bern, David Eppstein.
        Optimal Möbius Transformations for Information Visualization and Meshing.
        2001
        https://arxiv.org/abs/cs.CG/0101006
    ..[2] David Eppstein.
        Quasiconvex Programming.
        2004
        https://arxiv.org/abs/cs/0412046
    """
    viewpoint = np.array([0., 0., 0.])
    while True:
        new_spheres, r_min = transform_spheres(spheres=spheres,
                                               viewpoint=viewpoint,
                                               _return_spheres=True,
                                               _calc_grad=True,
                                               _return_value=True)
        grads = np.stack([new_sphere['gradient'] for new_sphere in new_spheres.values()
                          if abs(new_sphere['radius'] - r_min) < epsilon])
        print("r_min: %1.20f, viewpoint: %s" % (r_min, viewpoint))
        if grads.shape[0] == 1:
            d = grads[0]
        else:
            prb = pulp.LpProblem(name="search_direction",
                                 sense=pulp.LpMaximize)
            x, y, z = pulp.LpVariable('x'), pulp.LpVariable('y'), pulp.LpVariable('z')
            sums = np.sum(grads, axis=0)
            prb += sums[0] * x + sums[1] * y + sums[2] * z
            for i in range(grads.shape[0]):
                grad = grads[i]
                prb += grad[0] * x + grad[1] * y + grad[2] * z >= 0
                prb += grad[0] * x + grad[1] * y + grad[2] * z <= lg.norm(grad, ord=2)
            prb.solve()
            d = np.array([x.value(), y.value(), z.value()])
        if np.allclose(d, np.array([0., 0., 0.])):
            for new_sphere in new_spheres.values():
                new_sphere.pop('gradient')
            return new_spheres, viewpoint
        else:
            d /= lg.norm(d, ord=2)
            a_0, a_1, a_l, a_r = 0.0, (1 - lg.norm(viewpoint, ord=2)) / 2, None, None
            r_min_l, r_min_r = None, None
            for _ in range(100):
                if a_l is None:
                    a_l = (phi * a_0 + a_1) / (1 + phi)
                    r_min_l = transform_spheres(spheres=spheres,
                                                viewpoint=viewpoint + a_l * d,
                                                _return_spheres=False,
                                                _return_value=True)
                if a_r is None:
                    a_r = (a_0 + phi * a_1) / (phi + 1)
                    r_min_r = transform_spheres(spheres=spheres,
                                                viewpoint=viewpoint + a_r * d,
                                                _return_spheres=False,
                                                _return_value=True)
                if r_min_l > r_min_r:
                    a_1 = a_r
                    a_r = a_l
                    r_min_r = r_min_l
                    a_l = None
                else:
                    a_0 = a_l
                    a_l = a_r
                    r_min_l = r_min_r
                    a_r = None
            viewpoint += a_0 * d


def graph_to_poly(
    graph=None,
    primal=None,
    show_dual=True,
    epsilon=0.0000001
):
    graph, primal, primal_cpack, dual, dual_cpack = \
        circle_pack(graph=graph,
                    primal=primal,
                    epsilon=epsilon)
    primal_spheres = disks_to_spheres(disks=primal_cpack)
    print("\n========== Find Optimal Viewpoint ==========")
    new_primal_spheres, viewpoint = \
        maximize_minimum_sphere(spheres=primal_spheres,
                                epsilon=epsilon)

    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    plt.figaspect(1)
    print("\n========== Coordinates of Primal Nodes ==========")
    for v in primal.nodes:
        print("%s: %s" % (v, new_primal_spheres[v]['center']))
    centers = np.array([sphere['center'] for sphere in new_primal_spheres.values()]).T
    ax.scatter(centers[0], centers[1], centers[2], s=10, c="blue")
    for u, v in graph.edges.keys():
        cnt_u = new_primal_spheres[u]['center']
        cnt_v = new_primal_spheres[v]['center']
        ax.plot([cnt_u[0], cnt_v[0]],
                [cnt_u[1], cnt_v[1]],
                [cnt_u[2], cnt_v[2]],
                color='blue',
                linestyle='-',
                linewidth=1.0)

    if show_dual:
        dual_spheres = disks_to_spheres(disks=dual_cpack)
        new_dual_spheres = transform_spheres(spheres=dual_spheres,
                                             viewpoint=viewpoint,
                                             _return_spheres=True,
                                             _calc_grad=False,
                                             _return_value=False)
        print("\n========== Coordinates of Dual Nodes ==========")
        for v in primal.nodes:
            print("%s: %s" % (v, new_primal_spheres[v]['center']))
        centers = np.array([sphere['center'] for sphere in new_dual_spheres.values()]).T
        ax.scatter(centers[0], centers[1], centers[2], s=10, c="red")
        for u, v in dual.to_undirected().edges.keys():
            cnt_u = new_dual_spheres[u]['center']
            cnt_v = new_dual_spheres[v]['center']
            ax.plot([cnt_u[0], cnt_v[0]],
                    [cnt_u[1], cnt_v[1]],
                    [cnt_u[2], cnt_v[2]],
                    color='red',
                    linestyle='-',
                    linewidth=1.0)
    plt.show()


if __name__ == "__main__":

    graph = nx.tutte_graph()
    graph_to_poly(graph=graph,
                  primal=None,
                  show_dual=True)

    '''
    primal = nx.PlanarEmbedding()

    # Clockwize (cw) rotation around 0.
    primal.add_half_edge_cw(0, 1, None)
    primal.add_half_edge_cw(0, 3, 1)
    primal.add_half_edge_cw(0, 7, 3)
    primal.add_half_edge_cw(0, 4, 7)

    # Clockwise (cw) rotation around 1.
    primal.add_half_edge_cw(1, 0, None)
    primal.add_half_edge_cw(1, 5, 0)
    primal.add_half_edge_cw(1, 2, 5)

    # Clockwise (cw) rotation around 2.
    primal.add_half_edge_cw(2, 1, None)
    primal.add_half_edge_cw(2, 5, 1)
    primal.add_half_edge_cw(2, 6, 5)
    primal.add_half_edge_cw(2, 3, 6)

    # Clockwise (cw) rotation around 3.
    primal.add_half_edge_cw(3, 0, None)
    primal.add_half_edge_cw(3, 2, 0)
    primal.add_half_edge_cw(3, 7, 2)

    # Clockwise (cw) rotation around 4.
    primal.add_half_edge_cw(4, 0, None)
    primal.add_half_edge_cw(4, 7, 0)
    primal.add_half_edge_cw(4, 5, 7)

    # Clockwise (cw) rotation around 5.
    primal.add_half_edge_cw(5, 1, None)
    primal.add_half_edge_cw(5, 4, 1)
    primal.add_half_edge_cw(5, 6, 4)
    primal.add_half_edge_cw(5, 2, 6)

    # Clockwise (cw) rotation around 6.
    primal.add_half_edge_cw(6, 2, None)
    primal.add_half_edge_cw(6, 5, 2)
    primal.add_half_edge_cw(6, 7, 5)

    # Clockwise (cw) rotation around 7.
    primal.add_half_edge_cw(7, 0, None)
    primal.add_half_edge_cw(7, 3, 0)
    primal.add_half_edge_cw(7, 6, 3)
    primal.add_half_edge_cw(7, 4, 6)

    primal.check_structure()
    graph_to_poly(graph=None,
                  primal=primal,
                  show_dual=True)
    '''