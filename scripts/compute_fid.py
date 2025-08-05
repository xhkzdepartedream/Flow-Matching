from torch_fidelity import calculate_metrics

metrics = calculate_metrics(
    input1="../images_cifar10/",
input2="../samples_cifar10",
    cuda=True,
    fid=True,
    verbose=True,
)

print("FID:", metrics['frechet_inception_distance'])
