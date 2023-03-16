using Flux
using HDF5
using Random
using Flux: flatten
using Flux.Data: DataLoader
using ProgressBars
using Flux: update!
using Plots
using LinearAlgebra

# Load in the data
data_fn = "/scratch/smkatz/class/CS231A/E16Data/downsampled_64_mb.h5"
imgs = Float32.(h5read(data_fn, "X_train")) 
labels = Float32.(h5read(data_fn, "y_train"))

# Shuffle
Random.seed!(32)
shuffle_inds = randperm(size(imgs, 3))
imgs = imgs[:, :, shuffle_inds]
labels = labels[:, shuffle_inds]

# Split into train and valid
n_train = 9500
X_train = imgs[:, :, 1:n_train]
X_valid = imgs[:, :, n_train+1:end]
y_train = labels[1:4, 1:n_train]
y_valid = labels[1:4, n_train+1:end]

# Add channel dimension
X_train = reshape(X_train, size(X_train, 1), size(X_train, 2), 1, size(X_train, 3))
X_valid = reshape(X_valid, size(X_valid, 1), size(X_valid, 2), 1, size(X_valid, 3))

maximum(y_train, dims=2)

# Create dataloaders
device = gpu
train_dataloader = DataLoader((X_train |> device, y_train |> device), batchsize=32)
valid_dataloader = DataLoader((X_valid |> device, y_valid |> device), batchsize=32)

# Create model
function LeNet5(;imgsize = (64,64,1), out_dim = 4) 

    out_conv_size = (imgsize[1]÷4 - 3, imgsize[2]÷4 - 3, 16)
    
    return Chain(
            Conv((5, 5), imgsize[end]=>6, relu),
            MaxPool((2, 2)),
            Conv((5, 5), 6=>16, relu),
            MaxPool((2, 2)),
            flatten,
            Dense(prod(out_conv_size), 120, relu), 
            Dense(120, 84, relu), 
            Dense(84, out_dim)
          )
end

model = LeNet5()
model = model |> device

mse_loss(model, x, y) = Flux.Losses.mse(model(x), y)

θ = Flux.params(model)
opt = ADAM(1e-3)

nepoch = 200
iter = ProgressBar(1:nepoch)
for i in iter
    loss_tot = 0
    n_tot = 0
    for (x, y) in train_dataloader
        loss, back = Flux.pullback(() -> mse_loss(model, x, y), θ)
        update!(opt, θ, back(1f0))

        loss_tot += loss
        n_tot += 1
    end
    loss_val_tot = 0
    n_val_tot = 0
    for (x, y) in valid_dataloader
        loss_val_tot += mse_loss(model, x, y)
        n_val_tot += 1
    end
    set_postfix(iter, Loss = "$(loss_tot / n_tot)", LossVal = "$(loss_val_tot / n_val_tot)")
end

# Try to predict lines for first training image
img = X_train[:, :, 1, 1]
heatmap(Gray.(img'))

label = y_train[:, 1]

function get_pixels(m, b, w, h)
    xs = collect(1:w)
    ys = [m * x + b for x in xs]
    keep_inds = (ys .> 0) .& (ys .< h)
    return xs[keep_inds], ys[keep_inds]
end

h, w = size(img)
xs, ys = get_pixels(label[3], label[4], w, h)

p = heatmap(Gray.(img'))
plot!(p, xs, ys, color=:limegreen, lw=3, legend=false)

model = model |> cpu

preds = model(X_train[:, :, :, 1:10])
pred_label = preds[:, 1]

p = heatmap(Gray.(img'))
xs_right, ys_right = get_pixels(pred_label[1], pred_label[2], w, h)
xs_left, ys_left = get_pixels(pred_label[3], pred_label[4], w, h)
plot!(p, xs_right, ys_right, color=:limegreen, lw=3, legend=false)
plot!(p, xs_left, ys_left, color=:limegreen, lw=3, legend=false)

# Code for computing position and heading
####### Constants #######
K0 = [1144.08348083 0.00 -960. 0.0;
      0.0 -1144.08333778 -540.0 0.0;
      0.0 0.0 -1.0 0.0]
ground_plane = [21.05308266, -9970.33465342, 112.80709762, -25999.30320831]
u1 = [-0.00040965, -0.01131441, -0.99993591]
c1 = [0.3619336, -9.15347138, -578.61154869]

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
    det = sum(p3_normal.^2)

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
    return [cos(θ) 0.0 sin(θ);
            0.0 1.0 0.0;
           -sin(θ) 0.0 cos(θ)]
end

function get_distance(p1, l1, p2)
    n = l1 / norm(l1)
    m = cross(p1, n)

    d = norm(cross(p2, n) - m)

    return d
end

function get_state(pixel_line, K0, ground_plane, u1, c1; right=true)
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
    crosstrack = 11.417965507841265 - get_distance(p2rot, u2rot, c1)
    if !right
        crosstrack = -crosstrack
    end

    return crosstrack, rad2deg(heading)
end

pixel_line = undo_downsampling(label[1], label[2])
crosstrack, heading = get_state(pixel_line, K0, ground_plane, u1, c1)
labels[:, 1]

ind = 4
img = X_train[:, :, 1, ind]
heatmap(Gray.(img'))
label = labels[:, ind]
pred_label = preds[:, ind]

p = heatmap(Gray.(img'))
xs_right_true, ys_right_true = get_pixels(label[1], label[2], w, h)
xs_left_true, ys_left_true = get_pixels(label[3], label[4], w, h)
xs_right, ys_right = get_pixels(pred_label[1], pred_label[2], w, h)
xs_left, ys_left = get_pixels(pred_label[3], pred_label[4], w, h)
plot!(p, xs_right_true, ys_right_true, color=:cyan, lw=3, legend=false)
plot!(p, xs_left_true, ys_left_true, color=:cyan, lw=3, legend=false)
plot!(p, xs_right, ys_right, color=:limegreen, lw=3, legend=false)
plot!(p, xs_left, ys_left, color=:limegreen, lw=3, legend=false)

pixel_line_right = undo_downsampling(pred_label[1], pred_label[2])
pixel_line_left = undo_downsampling(pred_label[3], pred_label[4])
crosstrack_right, heading_right = get_state(pixel_line_right, K0, ground_plane, u1, c1)
crosstrack_left, heading_left = get_state(pixel_line_left, K0, ground_plane, u1, c1, right=false)

println("True Crosstrack: ", label[5])
println("Right Prediction: ", crosstrack_right)
println("Left Prediction: ", crosstrack_left)

println("True Heading: ", label[6])
println("Right Prediction: ", heading_right)
println("Left Prediction: ", heading_left)