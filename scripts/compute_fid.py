from torch_fidelity import calculate_metrics

metrics = calculate_metrics(
    input1="data/celeba_hq_256/",
    input2="celeba_samples/",
    cuda=True,
    fid=True,
    verbose=True,
)

print("FID:", metrics['frechet_inception_distance'])
