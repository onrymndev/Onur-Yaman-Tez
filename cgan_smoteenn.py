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

class ConditionalGenerator(nn.Module):
    def __init__(self, z_dim, label_dim, im_dim, hidden_dim=128):
        super().__init__()
        self.gen = nn.Sequential(
            get_generator_block(z_dim + label_dim, hidden_dim),
            get_generator_block(hidden_dim, hidden_dim * 2),
            get_generator_block(hidden_dim * 2, hidden_dim * 4),
            get_generator_block(hidden_dim * 4, hidden_dim * 8),
            nn.Linear(hidden_dim * 8, im_dim),
            nn.Sigmoid()
        )

    def forward(self, noise, labels):
        input = torch.cat((noise, labels), dim=1)
        return self.gen(input)

class ConditionalDiscriminator(nn.Module):
    def __init__(self, im_dim, label_dim, hidden_dim=128):
        super().__init__()
        self.disc = nn.Sequential(
            nn.Linear(im_dim + label_dim, hidden_dim * 4),
            nn.LeakyReLU(0.2),
            nn.Linear(hidden_dim * 4, hidden_dim * 2),
            nn.LeakyReLU(0.2),
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.Linear(hidden_dim, 1)
        )

    def forward(self, x, labels):
        input = torch.cat((x, labels), dim=1)
        return self.disc(input)

def get_disc_loss(gen, disc, criterion, real, real_labels, device, noise_labels, z_dim):
    batch_size = real.size(0)
    z = torch.randn(batch_size, z_dim).to(device)  # pure latent noise
    fake = gen(z, noise_labels)
    disc_fake_pred = disc(fake.detach(), noise_labels)
    disc_fake_loss = criterion(disc_fake_pred, torch.zeros_like(disc_fake_pred))
    disc_real_pred = disc(real, real_labels)
    disc_real_loss = criterion(disc_real_pred, torch.ones_like(disc_real_pred))
    return (disc_fake_loss + disc_real_loss) / 2

def get_gen_loss(gen, disc, criterion, batch_size, z_dim, device, labels):
    z = torch.randn(batch_size, z_dim).to(device)  # pure latent noise
    fake = gen(z, labels)
    disc_fake_pred = disc(fake, labels)
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

def cgan_augment_with_enn(X_train, y_train,apply_enn,
                                        n_epochs=200, batch_size=128, lr=0.00001 ):
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    feature_dim = X_train.shape[1]
    z_dim = feature_dim  # latent noise dimension can be tuned

    y_tr = y_train.ravel()
    # Select minority class real samples (assumed minority label = 1)
    X_real = np.array([X_train[i] for i in range(len(y_tr)) if int(y_tr[i]) == 1])
    y_real = np.ones(len(X_real))

    tensor_x = torch.Tensor(X_real)
    tensor_y = torch.Tensor(y_real).unsqueeze(1)  # shape (N,1)
    dataset = TensorDataset(tensor_x, tensor_y)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    label_dim = 1
    gen = ConditionalGenerator(z_dim=z_dim, label_dim=label_dim, im_dim=feature_dim).to(device)
    disc = ConditionalDiscriminator(im_dim=feature_dim, label_dim=label_dim).to(device)
    gen_opt = torch.optim.Adam(gen.parameters(), lr=lr)
    disc_opt = torch.optim.Adam(disc.parameters(), lr=lr)
    criterion = nn.BCEWithLogitsLoss()

    for epoch in range(n_epochs):
        for real, _ in tqdm(dataloader):
            cur_batch_size = len(real)
            real = real.to(device)
            real_labels = torch.ones((cur_batch_size, 1), device=device)  # real samples labeled 1
            noise_labels = torch.ones((cur_batch_size, 1), device=device)  # conditional label = 1 for generated

            disc_opt.zero_grad()
            disc_loss = get_disc_loss(gen, disc, criterion, real, real_labels, device, noise_labels, z_dim)
            disc_loss.backward()
            disc_opt.step()

            gen_opt.zero_grad()
            gen_loss = get_gen_loss(gen, disc, criterion, cur_batch_size, z_dim, device, noise_labels)
            gen_loss.backward()
            gen_opt.step()

        print(f"Epoch {epoch} | Gen Loss: {gen_loss.item():.4f}, Disc Loss: {disc_loss.item():.4f}")

    # Generate synthetic minority samples only from latent noise
    with torch.no_grad():
        n_samples = 2 * len(X_real)  # or any number you want
        noise = torch.randn(n_samples, z_dim).to(device)
        noise_labels = torch.ones((n_samples, 1), device=device)
        generated = gen(noise, noise_labels).cpu().numpy()

    # Combine real data + generated synthetic minority data
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