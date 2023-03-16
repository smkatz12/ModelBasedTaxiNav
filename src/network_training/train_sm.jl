using Flux
using HDF5
using Random
using Flux: flatten
using Flux.Data: DataLoader
using ProgressBars
using Flux: update!
using Plots
using LinearAlgebra
using Statistics

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

μ = mean(y_train, dims=2)
σ = std(y_train, dims=2)

y_train = (y_train .- μ) ./ σ
y_valid = (y_valid .- μ) ./ σ

# Add channel dimension
X_train = reshape(X_train, size(X_train, 1), size(X_train, 2), 1, size(X_train, 3))
X_valid = reshape(X_valid, size(X_valid, 1), size(X_valid, 2), 1, size(X_valid, 3))

# Create dataloaders
device = gpu
train_dataloader = DataLoader((X_train |> device, y_train |> device), batchsize=32)
valid_dataloader = DataLoader((X_valid |> device, y_valid |> device), batchsize=32)

# Create model
function LeNet5(; imgsize=(64, 64, 1), out_dim=4)

    out_conv_size = (imgsize[1] ÷ 4 - 3, imgsize[2] ÷ 4 - 3, 16)

    return Chain(
        Conv((5, 5), imgsize[end] => 6, relu),
        MaxPool((2, 2)),
        Conv((5, 5), 6 => 16, relu),
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
        update!(opt, θ, back(1.0f0))

        loss_tot += loss
        n_tot += 1
    end
    loss_val_tot = 0
    n_val_tot = 0
    for (x, y) in valid_dataloader
        loss_val_tot += mse_loss(model, x, y)
        n_val_tot += 1
    end
    set_postfix(iter, Loss="$(loss_tot / n_tot)", LossVal="$(loss_val_tot / n_val_tot)")
end

model = model |> cpu

using BSON: @save
@save "model_sm.bson" model μ σ