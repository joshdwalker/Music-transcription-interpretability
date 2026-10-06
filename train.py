import torch
from model import Autoencoder

def training_loop(
        autoencoder,
        training_activations_iterator,
        loss_function,
        learning_rate,
        eps=6.25e-10,
        clip_gradients=None,
        ema_multiplier=0.999
):
    scaler = torch.amp.GradScaler('cuda')
    autocast_ctx_manager = torch.amp.autocast('cuda')
    
    optimizer = torch.optim.Adam(autoencoder.parameters(), lr=learning_rate, eps=eps, fused=True)

    for i, flat_activations_training_batch in enumerate(training_activations_iterator):
        flat_activations_training_batch = flat_activations_training_batch.cuda()
        optimizer.zero_grad(set_to_none=True)

        with autocast_ctx_manager:
            reconstructions, info = autoencoder(flat_activations_training_batch)

            loss = loss_function(autoencoder, flat_activations_training_batch, reconstructions, info)

        print(i, loss.item())

        loss = scaler.scale(loss)
        loss.backward()
        
        # Unit normalize the autoencoder decoder weights
        autoencoder.decoder.weight.data /= autoencoder.decoder.weight.data.norm(dim=0)

        if clip_gradients is not None:
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(autoencoder.parameters(), clip_gradients)

        scaler.step(optimizer)
        scaler.update()

def main():
    # Load the model
    model = Autoencoder(input_dimension=768, latent_dimension=50000)