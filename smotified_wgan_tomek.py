import numpy as np
import torch
from torch import nn, autograd
from torch.utils.data import TensorDataset, DataLoader
from tqdm.auto import tqdm
from imblearn.under_sampling import TomekLinks

# ==== Generator ====
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
        )

    def forward(self, noise, labels):
        input = torch.cat((noise, labels), dim=1)
        return self.gen(input)

# ==== Discriminator ====
class ConditionalCritic(nn.Module):
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

# ==== Gradient Penalty ====
def gradient_penalty(critic, real_data, fake_data, labels, device, lambda_gp=10):
    batch_size = real_data.size(0)
    alpha = torch.rand(batch_size, 1).to(device)
    alpha = alpha.expand_as(real_data)

    interpolated = alpha * real_data + (1 - alpha) * fake_data
    interpolated = interpolated.to(device).requires_grad_(True)
    interpolated_labels = labels.to(device)

    critic_interpolated = critic(interpolated, interpolated_labels)

    gradients = autograd.grad(
        outputs=critic_interpolated,
        inputs=interpolated,
        grad_outputs=torch.ones_like(critic_interpolated),
        create_graph=True,
        retain_graph=True,
    )[0]

    gradients = gradients.view(batch_size, -1)
    gp = ((gradients.norm(2, dim=1) - 1) ** 2).mean() * lambda_gp
    return gp

# ==== Training ====
def shuffle_in_unison(a, b):
    assert len(a) == len(b)
    shuffled_a = np.empty(a.shape, dtype=a.dtype)
    shuffled_b = np.empty(b.shape, dtype=b.dtype)
    permutation = np.random.permutation(len(a))
    for old_index, new_index in enumerate(permutation):
        shuffled_a[new_index] = a[old_index]
        shuffled_b[new_index] = b[old_index]
    return shuffled_a, shuffled_b

def wgan_augment_with_existing_resampling(X_train, y_train, X_train_resampled, y_train_resampled,apply_tomek,
                    n_epochs=200, batch_size=128, critic_iterations=5, lr=1e-4):
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
    critic = ConditionalCritic(im_dim=input_dim, label_dim=label_dim).to(device)
    gen_opt = torch.optim.Adam(gen.parameters(), lr=lr, betas=(0.0, 0.9))
    critic_opt = torch.optim.Adam(critic.parameters(), lr=lr, betas=(0.0, 0.9))

    for epoch in range(n_epochs):
        for real, _ in tqdm(dataloader):
            real = real.to(device)
            real_labels = torch.ones((real.size(0), 1)).to(device)

            for _ in range(critic_iterations):
                z = torch.randn(real.size(0), input_dim).to(device)
                noise_labels = torch.ones((real.size(0), 1)).to(device)
                fake = gen(z, noise_labels)

                critic_real = critic(real, real_labels).mean()
                critic_fake = critic(fake.detach(), noise_labels).mean()
                gp = gradient_penalty(critic, real, fake.detach(), real_labels, device)

                critic_loss = -(critic_real - critic_fake) + gp
                critic_opt.zero_grad()
                critic_loss.backward()
                critic_opt.step()

            z = torch.randn(real.size(0), input_dim).to(device)
            noise_labels = torch.ones((real.size(0), 1)).to(device)
            gen_fake = gen(z, noise_labels)
            gen_loss = -critic(gen_fake, noise_labels).mean()

            gen_opt.zero_grad()
            gen_loss.backward()
            gen_opt.step()

        print(f"Epoch {epoch} | Gen Loss: {gen_loss.item():.4f}, Critic Loss: {critic_loss.item():.4f}")

    # Generate final samples
    with torch.no_grad():
        z = torch.randn(X_oversampled.shape[0], input_dim).to(device)
        noise_labels = torch.ones((X_oversampled.shape[0], 1)).to(device)
        generated = gen(z, noise_labels).cpu().numpy()

    final_data = np.concatenate((X_train_resampled[:t2[0]], generated), axis=0)
    final_labels = np.concatenate((y_train_resampled[:t2[0]], np.ones(len(generated))), axis=0)
    X_out, y_out= shuffle_in_unison(final_data, final_labels)

    if apply_tomek:
        tomek = TomekLinks()
        X_out, y_out = tomek.fit_resample(X_out, y_out)
   
    
    return X_out, y_out
