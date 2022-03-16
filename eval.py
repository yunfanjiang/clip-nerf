import os

import hydra
import torch
import clip
import numpy as np
from PIL import Image
from tqdm import tqdm


def run_eval(
    clip_model: str,
    generated_img_folder: str,
    generated_img_file_template: str,
    generated_img_start: int,
    generated_img_end: int,
    ground_truth_folder: str,
    ground_truth_file_template: str,
    ground_truth_start: int,
    ground_truth_end: int,
    text_prompt: str,
):
    # metric
    cos_sim = torch.nn.CosineSimilarity()

    generated_img_files = [
        generated_img_file_template.format(i)
        for i in range(generated_img_start, generated_img_end + 1)
    ]
    for file in generated_img_files:
        try:
            assert os.path.exists(os.path.join(generated_img_folder, file))
        except:
            print(os.path.join(generated_img_folder, file))
            exit()
    ground_truth_files = [
        ground_truth_file_template.format(i)
        for i in range(ground_truth_start, ground_truth_end + 1)
    ]
    for file in ground_truth_files:
        assert os.path.exists(os.path.join(ground_truth_folder, file))

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model, preprocess = clip.load(clip_model, device=device)

    # ground truth embeddings
    ground_truth_embds = []
    for file in tqdm(ground_truth_files):
        img = (
            preprocess(Image.open(os.path.join(ground_truth_folder, file)))
            .unsqueeze(0)
            .to(device)
        )
        with torch.no_grad():
            ground_truth_embds.append(model.encode_image(img))
    ground_truth_embds = torch.concat(ground_truth_embds, dim=0)

    # generated imgs embeddings
    generated_embds = []
    for file in tqdm(generated_img_files):
        img = (
            preprocess(Image.open(os.path.join(generated_img_folder, file)))
            .unsqueeze(0)
            .to(device)
        )
        with torch.no_grad():
            generated_embds.append(model.encode_image(img))
    generated_embds = torch.concat(generated_embds, dim=0)

    assert ground_truth_embds.shape[1] == generated_embds.shape[1]

    # text embedding
    text = clip.tokenize([text_prompt]).to(device)
    with torch.no_grad():
        text_embd = model.encode_text(text)

    # score ground truth
    with torch.no_grad():
        gt_score = cos_sim(ground_truth_embds, text_embd)
    gt_score = gt_score.cpu().numpy()

    # score generated
    with torch.no_grad():
        generated_score = cos_sim(generated_embds, text_embd)
    generated_score = generated_score.cpu().numpy()

    print(f"\n Ground-truth score mean: {np.mean(gt_score)}, std: {np.std(gt_score)}\n Generated score mean: {np.mean(generated_score)}, std: {np.std(generated_score)}")


@hydra.main(config_path="./eval_cfg", config_name="conf")
def main(cfg):
    run_eval(**cfg.eval)


if __name__ == "__main__":
    main()
