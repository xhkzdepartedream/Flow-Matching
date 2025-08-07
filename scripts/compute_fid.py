from torch_fidelity import calculate_metrics

metrics = calculate_metrics(
    input1="/data1/yangyanliang/.cache/kagglehub/datasets/badasstechie/celebahq-resized-256x256/versions/1/celeba_hq_256/",
    input2="/data1/yangyanliang/Flow-Matching/celeba_samples/",
    cuda=True,
    fid=True,
    verbose=True,
)

print("FID:", metrics['frechet_inception_distance'])
