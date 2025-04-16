import os
from tqdm import tqdm
import torch
from torch.utils.data import DataLoader
from torch.optim import Adam

from lm_steer.arguments import parse_args
from lm_steer.models.get_model import get_model
from lm_steer.utils import RunningMean

from data import load_dataset


def main(args):
    train_data = load_dataset(args.dataset_name, args.data_dir, args.subset)
    dataloader = DataLoader(
        train_data, batch_size=args.batch_size,
        # 打乱数据
        shuffle=True)
    data_iter = iter(dataloader)

    device = torch.device("cuda:0") if args.cuda else torch.device("cpu")
    model, tokenizer = get_model(
        args.model_name, args.adapted_component, args.adaptor_class,
        args.num_steers,
        args.rank, args.epsilon, args.init_var, args.low_resource_mode)
    model.to_device(device)

    print("number of training steps:", args.n_steps)
    start_step = 0
    if os.path.exists(args.ckpt_name):
        try:
            ckpt = torch.load(args.ckpt_name, weights_only=False)
        except:
            from argparse import Namespace
            torch.serialization.add_safe_globals([Namespace])
            ckpt = torch.load(args.ckpt_name, weights_only=True)
        model.load_state_dict(ckpt[1])
        # Reset start_step to 0 to ensure full training
        start_step = 0
        print(f"Loaded checkpoint, starting fresh training from step 0")
        # 修改这里，确保 start_step 不会等于或超过 n_steps
        start_step = min(ckpt[2], args.n_steps - 1)
        print(f"resume training from {start_step}")
    if args.optimizer == "Adam":
        optimizer = Adam(model.parameters(), lr=args.lr)

    pbar = tqdm(range(start_step, args.n_steps))
    loss_mean = RunningMean(args.gamma_mean)
    # 更新 GradScaler 的使用方式
    scaler = torch.amp.GradScaler('cuda')

    for step_i in pbar:
        batch = next(data_iter, None)
        if batch is None:
            data_iter = iter(dataloader)
            batch = next(data_iter, None)

        cur_batch_size = len(batch["text"])
        batch_stance = torch.zeros(cur_batch_size, args.num_steers).to(device)
        batch_stance[:, args.training_steer] = torch.Tensor(
            batch["label"]).to(device)
        if args.dummy_steer is not None:
            batch_stance[:, args.dummy_steer] = 1
        batch_text = batch["text"]
        tokenized = tokenizer(batch_text, padding=True,
                              max_length=args.max_length, truncation=True)
        input_ids = torch.LongTensor(tokenized["input_ids"]).to(device)

        optimizer.zero_grad()
        attention_mask = torch.LongTensor(tokenized["attention_mask"]).to(
            device)
        if args.low_resource_mode:
            with torch.amp.autocast(
                device_type="cuda", dtype=torch.float16
            ):
                loss = model(
                    input_ids, attention_mask,
                    batch_stance.float()
                ).loss
                regularization_term = model.regularization_term()
            scaler.scale(loss + args.regularization * regularization_term
                         ).backward()
            scaler.step(optimizer)
            scaler.update()
        else:
            loss = model(
                input_ids, attention_mask,
                batch_stance.float()
            ).loss
            regularization_term = model.regularization_term()
            (loss + args.regularization * regularization_term).backward()
            optimizer.step()

        loss_mean.update(loss)
        pbar.set_description(
            f"{loss_mean.value}, {regularization_term.item()}")
        if (step_i+1) % args.log_step == 0:
            print(pbar.desc, flush=True)

    torch.save([
        args, model.state_dict(),
        max(args.n_steps, start_step)
    ], args.ckpt_name)


if __name__ == "__main__":
    args = parse_args()
    main(args)
