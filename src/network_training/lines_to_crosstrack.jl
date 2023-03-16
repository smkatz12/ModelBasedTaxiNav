using LinearAlgebra

####### Constants #######
K0 = [1144.08348083 0.00 -960.0 0.0
    0.0 -1144.08333778 -540.0 0.0
    0.0 0.0 -1.0 0.0]

ground_plane_sm = [21.05308266, -9970.33465342, 112.80709762, -25999.30320831]
u1_sm = [-0.00040965, -0.01131441, -0.99993591]
c1_sm = [0.3619336, -9.15347138, -578.61154869]
hw_sm = 11.417965507841265 # Half Width

ground_plane_kmwh = [6.92491392, 18648.60891881, -61.77202323, 41670.59509746]
u1_kmwh = [-0.01113786, 0.00331633, 0.99993247]
c1_kmwh = [13.38632798, -6.11540012, -1170.11564816]
hw_kmwh = 13.0 # Half Width
#########################

function undo_downsampling(m, b)
    # Get two points on the line
    p1 = [0, b]
    p2 = [1, m + b]

    # Undownsample them
    p1_undo = [p1[1] * 30.0, p1[2] * 9.0 + 504.0, 1.0]
    p2_undo = [p2[1] * 30.0, p2[2] * 9.0 + 504.0, 1.0]

    # Get line
    line = cross(p1_undo, p2_undo)

    return line
end

function get_plane_int(p1, p2)
    p1_normal = p1[1:3]
    p2_normal = p2[1:3]

    p3_normal = cross(p1_normal, p2_normal)
    det = sum(p3_normal .^ 2)

    r_point = (cross(p3_normal, p2_normal) * p1[4] + cross(p1_normal, p3_normal) * p2[4]) / det
    r_normal = p3_normal

    return r_point, r_normal
end

function get_corresponding_plane(l, K0)
    return K0' * l
end

function get_3d_line(l, K0, ground_plane)
    edge_plane = get_corresponding_plane(l, K0)
    r_point, r_normal = get_plane_int(edge_plane, ground_plane)
    return r_point, r_normal
end

function get_rot_y(θ)
    return [cos(θ) 0.0 sin(θ)
        0.0 1.0 0.0
        -sin(θ) 0.0 cos(θ)]
end

function get_distance(p1, l1, p2)
    n = l1 / norm(l1)
    m = cross(p1, n)

    d = norm(cross(p2, n) - m)

    return d
end

function get_state(m, b, K0, ground_plane, u1, c1, hw; right=true)
    pixel_line = undo_downsampling(m, b)
    p2, u2 = get_3d_line(pixel_line, K0, ground_plane)

    # Determine heading
    cos_heading = dot(u1, u2) / (norm(u1) * norm(u2))
    if cos_heading > 1.0
        cos_heading = 1.0
    end
    heading = acos(cos_heading)

    # Unrotate
    if right
        Ry = get_rot_y(heading)
        p2rot, u2rot = Ry * p2, Ry * u2
        if norm(cross(u1, u2rot / norm(u2rot))) > 1e-2
            # Rotated the wrong way
            heading = -heading
            Ry = get_rot_y(heading)
            p2rot, u2rot = Ry * p2, Ry * u2
        end
    else
        heading = π - heading
        Ry = get_rot_y(heading)
        p2rot, u2rot = Ry * p2, Ry * u2
        if norm(cross(u1, u2rot / norm(u2rot))) > 1e-2
            # Rotated the wrong way
            heading = -heading
            Ry = get_rot_y(heading)
            p2rot, u2rot = Ry * p2, Ry * u2
        end
    end

    # Determine crosstrack
    crosstrack = hw - get_distance(p2rot, u2rot, c1)
    if !right
        crosstrack = -crosstrack
    end

    return crosstrack, rad2deg(heading)
end

function runway_width(m1, b1, m2, b2, K0, ground_plane)
    pixel_line = undo_downsampling(m1, b1)
    p21, u21 = get_3d_line(pixel_line, K0, ground_plane)

    pixel_line = undo_downsampling(m2, b2)
    p22, u22 = get_3d_line(pixel_line, K0, ground_plane)

    return get_distance(p21, u21, p22)
end