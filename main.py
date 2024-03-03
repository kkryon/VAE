import matplotlib.pyplot as plt
import torch
from data.load_data import Data
from model.model import VAE

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
data = Data()
train_loader, test_loader = data.get_dataloaders()
data.show_images(train_loader)


# trainer = L.Trainer(max_epochs=100)
# trainer.fit(model=VAE(), train_dataloaders=krain_loader)

model = VAE.load_from_checkpoint(
    "lightning_logs/version_4/checkpoints/epoch=99-step=5900.ckpt"
)


plt.imshow(
    model.forward(next(iter(test_loader))[0][5].to("cuda"))[1]
    .detach()
    .cpu()
    .numpy()
    .reshape(28, 28),
    cmap="gray",
    interpolation="none",
)
# plt.imshow(data.test_data[5].reshape(28, 28), cmap="gray", interpolation="none")
plt.xticks([])
plt.yticks([])
plt.savefig("data/sample.png")
