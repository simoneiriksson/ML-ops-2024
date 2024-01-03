import click
import torch
from torch import nn
from model import MyAwesomeModel
from model import MyAwesomeModelConv

from data import mnist

@click.group()
def cli():
    """Command line interface."""
    pass


@click.command()
@click.option("--lr", default=1e-3, help="learning rate to use for training")
@click.option("--outfile", default="trained_model.pt", help="Output file name")
def train(lr, outfile):
    torch.random.manual_seed(0)
    """Train a model on MNIST."""
    print("Training day and night")
    print(lr)
    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    device = "cpu"
    print(f"{device = }")
    # TODO: Implement training loop here
    model = MyAwesomeModel().to(device)
    criterion = nn.NLLLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    train_loader, _ = mnist()
    epochs = 10
    for epoch in range(epochs):
        loss_accum = 0  # to keep track of the loss value
        for images, labels in train_loader:
            #images = images.view(images.shape[0], -1).to(device)
            images = images.to(device)
            optimizer.zero_grad()
            output = model(images)
            loss = criterion(output, labels.to(device))
            loss.backward()
            optimizer.step()
            loss_accum += loss.item()
        print("Epoch: {}/{}.. ".format(epoch+1, epochs),
              "Training Loss: {:.3f}.. ".format(loss.item()))
    torch.save(model, outfile)

@click.command()
@click.argument("model_checkpoint")
def evaluate(model_checkpoint):
    """Evaluate a trained model."""
    print("Evaluating like my life dependends on it")
    print(model_checkpoint)

    # TODO: Implement evaluation logic here
    model = torch.load(model_checkpoint)
    _, testloader = mnist()
    with torch.no_grad():
        model.eval()
        num_equals = 0
        num_tot = 0
        for images, labels in testloader:
            # Get the class probabilities
            ps = torch.exp(model(images))
            top_p, top_class = ps.topk(1, dim=1)
            equals = top_class == labels.view(*top_class.shape)
            #print(f"{equals.shape = }")
            num_equals += equals.sum()
            num_tot += equals.shape[0]
            #print(f"{num_equals=}, {num_tot=}")
        ## TODO: Implement the validation pass and print out the validation accuracy
        accuracy = num_equals/num_tot
        print(f'Accuracy: {accuracy.item():0.2%}')

cli.add_command(train)
cli.add_command(evaluate)


if __name__ == "__main__":
    cli()
