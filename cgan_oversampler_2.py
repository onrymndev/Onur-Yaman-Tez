import numpy as np
import torch
from torch import nn
from torch.utils.data import TensorDataset, DataLoader
from tqdm.auto import tqdm

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

def get_disc_loss(gen, disc, criterion, real, real_labels, device, noise_labels):
    z = torch.randn(real.size(0), real.size(1)).to(device)
    fake = gen(z, noise_labels)
    disc_fake_pred = disc(fake.detach(), noise_labels)
    disc_fake_loss = criterion(disc_fake_pred, torch.zeros_like(disc_fake_pred))
    disc_real_pred = disc(real, real_labels)
    disc_real_loss = criterion(disc_real_pred, torch.ones_like(disc_real_pred))
    return (disc_fake_loss + disc_real_loss) / 2

def get_gen_loss(gen, disc, criterion, batch_size, input_dim, device, labels):
    z = torch.randn(batch_size, input_dim).to(device)
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

def cgan_augment_with_existing_resampling(X_train, y_train, X_train_resampled, y_train_resampled,
                                         n_epochs=200, batch_size=128, lr=0.00001):
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    input_dim = X_train.shape[1]
    t2 = X_train.shape
    X_oversampled = X_train_resampled[t2[0]:]
    X_oversampled = torch.from_numpy(X_oversampled.astype(np.float32))

    y_tr = y_train.ravel()
    X_real = np.array([X_train[i] for i in range(len(y_tr)) if int(y_tr[i]) == 1])
    y_real = np.ones(len(X_real))

    tensor_x = torch.Tensor(X_real)
    tensor_y = torch.Tensor(y_real)
    dataset = TensorDataset(tensor_x, tensor_y)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    label_dim = 1
    gen = ConditionalGenerator(z_dim=input_dim, label_dim=label_dim, im_dim=input_dim).to(device)
    disc = ConditionalDiscriminator(im_dim=input_dim, label_dim=label_dim).to(device)
    gen_opt = torch.optim.Adam(gen.parameters(), lr=lr)
    disc_opt = torch.optim.Adam(disc.parameters(), lr=lr)
    criterion = nn.BCEWithLogitsLoss()

    for epoch in range(n_epochs):
        for real, _ in tqdm(dataloader):
            cur_batch_size = len(real)
            real = real.to(device)

            # Create labels for class '1' (minority)
            real_labels = torch.ones((cur_batch_size, 1)).to(device)
            noise_labels = torch.ones((cur_batch_size, 1)).to(device)

            disc_opt.zero_grad()
            disc_loss = get_disc_loss(gen, disc, criterion, real, real_labels, device, noise_labels)
            disc_loss.backward()
            disc_opt.step()

            gen_opt.zero_grad()
            gen_loss = get_gen_loss(gen, disc, criterion, cur_batch_size, input_dim, device, noise_labels)
            gen_loss.backward()
            gen_opt.step()

        print(f"Epoch {epoch} | Gen Loss: {gen_loss.item():.4f}, Disc Loss: {disc_loss.item():.4f}")

    with torch.no_grad():
        noise = torch.randn(X_oversampled.shape[0], input_dim).to(device)
        noise_labels = torch.ones((X_oversampled.shape[0], 1)).to(device)
        generated = gen(noise, noise_labels).cpu().numpy()

    final_data = np.concatenate((X_train_resampled[:t2[0]], generated), axis=0)
    final_labels = np.concatenate((y_train_resampled[:t2[0]], np.ones(len(generated))), axis=0)

    Xn, yn = shuffle_in_unison(final_data, final_labels)
    return Xn, yn
