import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset
from tqdm import tqdm
from imblearn.under_sampling import TomekLinks
from sklearn.utils import shuffle as shuffle_in_unison
from imblearn.under_sampling import EditedNearestNeighbours
from adaptive_power_enn import adaptive_power_enn_fast

def get_generator_block(input_dim, output_dim):
    return nn.Sequential(
        nn.Linear(input_dim, output_dim),
        nn.BatchNorm1d(output_dim),
        nn.ReLU(inplace=True),
    )

class Generator(nn.Module):
    def __init__(self, z_dim, im_dim, hidden_dim=128):
        super(Generator, self).__init__()
        self.gen = nn.Sequential(
            get_generator_block(z_dim, hidden_dim),
            get_generator_block(hidden_dim, hidden_dim * 2),
            get_generator_block(hidden_dim * 2, hidden_dim * 4),
            get_generator_block(hidden_dim * 4, hidden_dim * 8),
            nn.Linear(hidden_dim * 8, im_dim),
            nn.Sigmoid()
        )

    def forward(self, noise):
        return self.gen(noise)

class Discriminator(nn.Module):
    def __init__(self, im_dim, hidden_dim=128):
        super(Discriminator, self).__init__()
        self.disc = nn.Sequential(
            nn.Linear(im_dim, hidden_dim * 4),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(hidden_dim * 4, hidden_dim * 2),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.Linear(hidden_dim, 1)
        )

    def forward(self, image):
        return self.disc(image)

def get_noise(n_samples, z_dim, device):
    """Generate random latent noise vectors."""
    return torch.randn(n_samples, z_dim, device=device)

def get_disc_loss(gen, disc, criterion, real, device, z_dim):
    batch_size = real.size(0)
    fake_noise = get_noise(batch_size, z_dim, device=device)
    fake = gen(fake_noise)
    disc_fake_pred = disc(fake.detach())
    disc_fake_loss = criterion(disc_fake_pred, torch.zeros_like(disc_fake_pred))
    disc_real_pred = disc(real)
    disc_real_loss = criterion(disc_real_pred, torch.ones_like(disc_real_pred))
    return (disc_fake_loss + disc_real_loss) / 2

def get_gen_loss(gen, disc, criterion, device, z_dim, batch_size):
    fake_noise = get_noise(batch_size, z_dim, device=device)
    fake = gen(fake_noise)
    disc_fake_pred = disc(fake)
    return criterion(disc_fake_pred, torch.ones_like(disc_fake_pred))

def shuffle_in_unison(a, b):
    assert len(a) == len(b)
    shuffled_a = np.empty(a.shape, dtype=a.dtype)
    shuffled_b = np.empty(b.shape, dtype=b.dtype)
    permutation = np.random.permutation(len(a))
    for old_index, new_index in enumerate(permutation):
        shuffled_a[new_index] = a[old_index]
        shuffled_b[new_index] = b[old_index]
    return shuffled_a, shuffled_b

def gan_augment_with_enn(X_train, y_train, apply_enn,
                                   n_epochs=200, batch_size=128, lr=0.00001):
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    input_dim = X_train.shape[1]

    # === 1️⃣ Extract minority class (label 1)
    y_tr = y_train.ravel()
    X_real = np.array([X_train[i] for i in range(len(y_tr)) if int(y_tr[i]) == 1])
    y_real = np.ones(len(X_real))

    tensor_x = torch.Tensor(X_real)
    tensor_y = torch.Tensor(y_real)
    dataset = TensorDataset(tensor_x, tensor_y)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    # === 2️⃣ Initialize models
    gen = Generator(z_dim=input_dim, im_dim=input_dim).to(device)
    disc = Discriminator(im_dim=input_dim).to(device)
    gen_opt = torch.optim.Adam(gen.parameters(), lr=lr)
    disc_opt = torch.optim.Adam(disc.parameters(), lr=lr)
    criterion = nn.BCEWithLogitsLoss()

    # === 3️⃣ GAN Training
    for epoch in range(n_epochs):
        for real, _ in tqdm(dataloader, leave=False):
            cur_batch_size = len(real)
            real = real.view(cur_batch_size, -1).to(device)

            # --- Discriminator step
            disc_opt.zero_grad()
            disc_loss = get_disc_loss(gen, disc, criterion, real, device, input_dim)
            disc_loss.backward(retain_graph=True)
            disc_opt.step()

            # --- Generator step
            gen_opt.zero_grad()
            gen_loss = get_gen_loss(gen, disc, criterion, device, input_dim, cur_batch_size)
            gen_loss.backward()
            gen_opt.step()

        print(f"Epoch {epoch} | Gen Loss: {gen_loss.item():.4f}, Disc Loss: {disc_loss.item():.4f}")

    # === 4️⃣ Generate synthetic samples with latent noise
    with torch.no_grad():
        n_generate = len(X_real) * 2  # You can adjust multiplier
        noise = get_noise(n_generate, input_dim, device=device)
        generated = gen(noise).cpu().numpy()

    # === 5️⃣ Combine original data and synthetic positives
    final_data = np.concatenate((X_train, generated), axis=0)
    final_labels = np.concatenate((y_train, np.ones(len(generated))), axis=0)

    X_out, y_out = shuffle_in_unison(final_data, final_labels)

    # === 6️⃣ Optional: Clean with Tomek Links
    if apply_enn:
        print(f"Before ENN: {X_out.shape}, Balance: {np.bincount(y_out.astype(int))}")
        enn = EditedNearestNeighbours()
        X_out, y_out = enn.fit_resample(X_out, y_out)
        print(f"After ENN: {X_out.shape}, Balance: {np.bincount(y_out.astype(int))}")

    return X_out, y_out