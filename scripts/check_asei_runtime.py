"""Small single-device predictor throughput and scan correctness/timing probe."""
import argparse
import json
import logging
import os
from pathlib import Path
import statistics
import sys
import time
import numpy as np
import torch

sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from ts_benchmark.baselines.self_impl import ASEI
from ts_benchmark.baselines.self_impl.ASEI.model import EvidenceScorer, Predictor, lag_graph
from ts_benchmark.baselines.self_impl.ASEI.progress import IMPLEMENTATION_VERSION


def check_recurrence(device,channels,steps):
    torch.manual_seed(17)
    features = torch.randn(2,2,steps,channels,4,device=device)
    records,outputs,gradients = [],[],[]
    for backend in ("reference","scan"):
        model = EvidenceScorer(6,.05,.8,.98,.9,recurrence_backend=backend).to(device)
        timings = []
        for iteration in range(4):
            model.zero_grad(set_to_none=True)
            if device.type=="cuda":
                torch.cuda.synchronize(device)
            started = time.perf_counter()
            scores,logits = model(features)
            (scores.square().mean()+logits.square().mean()).backward()
            if device.type=="cuda":
                torch.cuda.synchronize(device)
            if iteration:
                timings.append(time.perf_counter()-started)
        outputs.append(scores.detach().cpu())
        gradients.append([p.grad.detach().cpu().clone() for p in model.parameters()])
        records.append(dict(backend=backend,median_forward_backward_seconds=statistics.median(timings)))
    torch.testing.assert_close(*outputs,rtol=1e-5,atol=5e-5)
    for expected,actual in zip(*gradients):
        torch.testing.assert_close(actual,expected,rtol=1e-4,atol=5e-4)
    result = dict(check="recurrence",device=str(device),steps=steps,channels=channels,
                  output_max_abs_error=float((outputs[0]-outputs[1]).abs().max()),
                  gradient_max_abs_error=max(float((a-b).abs().max()) for a,b in zip(*gradients)),
                  timings=records,scope="Synthetic timing; not a benchmark speed claim")
    print(json.dumps(result),flush=True)
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--gpus",type=int,nargs=1,default=[1])
    parser.add_argument("--device",default="auto")
    parser.add_argument("--channels",type=int,default=51)
    parser.add_argument("--batch-size",type=int,default=8,help="Number of continuous target blocks")
    parser.add_argument("--win-size",type=int,default=63)
    parser.add_argument("--batches",type=int,default=2)
    parser.add_argument("--block-sizes",type=int,nargs="+",default=[128,256])
    parser.add_argument("--threads",type=int,default=4)
    parser.add_argument("--score-steps",type=int,default=1024)
    args = parser.parse_args()
    if min(args.channels,args.batch_size,args.win_size,args.batches,args.threads,args.score_steps,
           args.gpus[0]+1,*args.block_sizes)<1:
        parser.error("Counts must be positive and the GPU ID nonnegative")
    os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpus[0])
    logging.basicConfig(level=logging.INFO,format="%(asctime)s %(levelname)s %(message)s")
    torch.set_num_threads(args.threads)
    for block in args.block_sizes:
        model = ASEI(device=args.device,batch_size=args.batch_size,predictor_block_size=block,win_size=args.win_size)
        model._resolve_capacity(8192)
        count = args.batch_size*block*args.batches
        values = np.random.default_rng(42).normal(size=(model.history+count,args.channels)).astype(np.float32)
        cfg = model.config
        model.columns_ = list(range(args.channels))
        graph = lag_graph(values,cfg.graph_lags,cfg.graph_neighbors,cfg.graph_sample_cap)
        model.predictor = Predictor(args.channels,*graph,hidden=cfg.hidden_dim,embedding=cfg.embedding_dim,
                                    win_size=cfg.win_size).to(model.device)
        loader = model._predictor_loader(values,True)
        optimizer = torch.optim.AdamW(model.predictor.parameters(),lr=cfg.learning_rate,weight_decay=cfg.predictor_weight_decay)
        if model.device.type=="cuda":
            torch.cuda.synchronize(model.device)
        started = time.perf_counter()
        loss = model._predictor_loss(loader,optimizer,epoch="diagnostic")
        if model.device.type=="cuda":
            torch.cuda.synchronize(model.device)
        elapsed = time.perf_counter()-started
        print(json.dumps(dict(implementation=IMPLEMENTATION_VERSION,device=str(model.device),
                              GPU=torch.cuda.get_device_name(model.device) if model.device.type=="cuda" else None,
                              block_size=block,blocks_per_batch=cfg.batch_size,batches=len(loader),
                              target_points=count,seconds=elapsed,target_points_per_second=count/elapsed,loss=loss)),flush=True)
        device = model.device
        del loader,optimizer,model
    check_recurrence(device,args.channels,args.score_steps)


if __name__=="__main__":
    main()
