import argparse, time, torch, logging, os
import numpy as np
from datetime import datetime
from torch.optim.lr_scheduler import ExponentialLR, StepLR
from utils import fix_seed, get_mappings, get_main, calc_mrr, setup_logger
from model import MRCCA
from built_graph import DataLoader

def main(args):

    def eval_permit(cur_epoch):
        do_eval = False
        if cur_epoch >= args.start_test_at:
            do_eval = True
        return do_eval

    exec_name = datetime.today().strftime('%Y-%m-%d-%H-%M')+'-'+args.dataset
    os.makedirs('./log', exist_ok=True)
    os.makedirs('./cache', exist_ok=True)
    log_file_path = './log/'+exec_name+'.log'
    model_state_file = './cache/'+exec_name+'.pth'
    logger = setup_logger(name=exec_name, level=logging.INFO, log_file=log_file_path)
    logger.info(args)
    seed_value = 2411
    fix_seed(seed_value, random_lib=True, numpy_lib=True, torch_lib=True)
    # load graph data
    if args.dataset == 'fb15k-237':
        ds_dir_name = './data/FB15K-237/'
    elif args.dataset == 'wn18rr':
        ds_dir_name = './data/WN18RR/'
    elif args.dataset == 'Kinship':
        ds_dir_name = './data/Kinship/'
    # 实体和关系名字+id
    names2ids, rels2ids = get_mappings(
        [ds_dir_name + 'train.txt',
         ds_dir_name + 'valid.txt',
         ds_dir_name + 'test.txt']
    )
    #add graph
    data_loader = DataLoader(ds_dir_name, names2ids, rels2ids)

    train_data = get_main(
        ds_dir_name,
        'train.txt',
        names2ids,
        rels2ids,
        add_inverse=True
    )['triples']

    valid_data = get_main(
        ds_dir_name,
        'valid.txt',
        names2ids,
        rels2ids,
        add_inverse=False
    )['triples']

    test_data = get_main(
        ds_dir_name,
        'test.txt',
        names2ids,
        rels2ids,
        add_inverse=False
    )['triples']

    n_samples = train_data.shape[0] // 2
    batch_size = args.batch_size // 2
    packed_train_data = [
        np.vstack(
            (train_data[i * batch_size:min((i + 1) * batch_size, n_samples)],
             train_data[i * batch_size + n_samples:min((i + 1) * batch_size, n_samples) + n_samples])
        )
        for i in range(int(np.ceil(n_samples / batch_size)))
    ]

    logger.info(f'train shape: {train_data.shape}, valid shape: {valid_data.shape}, test shape: {test_data.shape}')
    logger.info(f'num entities: {len(names2ids)}, num relations: {len(rels2ids)}')

    num_nodes = len(names2ids.keys())
    num_rels = len(rels2ids.keys())

    # check cuda
    use_cuda = args.gpu >= 0 and torch.cuda.is_available()
    if use_cuda:
        torch.cuda.set_device(args.gpu)

    # create model
    model = MRCCA(
        num_nodes,
        num_rels,
        args.d_embed,
        args.d_k,
        args.d_model,
        args.d_inner,
        args.num_segment,
        args.MR_method,
        **{'dr_in': args.dr_in,
           'dr_hi': args.dr_hi,
           'dr_fm': args.dr_fm,
           }
    )

    valid_data = torch.LongTensor(valid_data)
    test_data = torch.LongTensor(test_data)
    if use_cuda:
        valid_data = valid_data.cuda()
        test_data = test_data.cuda()
        model.cuda()

    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    scheduler = StepLR(optimizer, step_size=args.lr_step_decay, gamma=args.lr_decay)
    # scheduler = ExponentialLR(optimizer, args.lr_decay)

    # training loop
    logger.info('start training...')
    best_mrr = 0
    best_epoch = 0
    for epoch in range(1, args.n_epochs + 1):
        for edges in packed_train_data:
            model.train()

            edges = torch.as_tensor(edges)

            if use_cuda:
                edges = edges.cuda()
                graph = data_loader.graph.cuda()
            scores = model(edges, graph)
            loss = model.cal_loss(scores, edges)
            loss.backward()
            # torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_norm)  # clip gradients

            optimizer.step()
            optimizer.zero_grad()

        if epoch <= args.decay_until:
            scheduler.step()

        if epoch % 5 == 0:
            logger.info(f'Epoch {epoch:04d} | Loss {loss.item():.7f} | Best MRR {best_mrr:.4f} | Best epoch {best_epoch:04d}')

        # validation
        if epoch % 20 == 0:
            if  eval_permit(epoch):
                with torch.no_grad():
                    model.eval()
                    logger.info('start eval')
                    mrr = calc_mrr(model, torch.LongTensor(train_data[:train_data.shape[0] // 2]).cuda(), valid_data, test_data,
                               graph, hits=[1, 3, 10], logger=logger)
                    # save best model
                    if best_mrr < mrr:
                        best_mrr = mrr
                        best_epoch = epoch
                        logger.info(f'Epoch {epoch:04d} | Loss {loss.item():.7f} | Best MRR {best_mrr:.4f} | Best epoch {best_epoch:04d}')
                        if epoch >= args.save_epochs:
                            torch.save({'state_dict': model.state_dict(), 'epoch': epoch}, model_state_file)

                    logger.info('eval done!')

    logger.info('training done!')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='MRCCA')
    parser.add_argument("--gpu", type=int, default=0,
                        help="gpu")
    parser.add_argument("--lr", type=float, default=1e-3,
                        help="learning rate")
    parser.add_argument("--lr-decay", type=float, default=0.995,
                        help="learning rate decay rate")
    parser.add_argument("--n-epochs", type=int, default=4500,
                        help="number of minimum training epochs")
    parser.add_argument("-d", "--dataset", type=str, default='fb15k-237',
                        help="dataset to use: fb15k-237 or wn18rr or Kinship")
    parser.add_argument("--batch-size", type=int, default=2048,
                        help="number of triples to sample at each iteration")
    parser.add_argument("--evaluate-every", type=int, default=20,
                        help="perform evaluation every n epochs")
    parser.add_argument("--start-test-at", type=int, default=300,
                        help="firs epoch to evaluate on test data for each epoch")
    parser.add_argument("--lr-step-decay", type=int, default=2,
                        help="decay lr every x steps")
    parser.add_argument("--save-epochs", type=int, default=6000,
                        help="save per epoch")
    parser.add_argument("--num-segment", type=int, default=64,
                        help="number of attention heads")
    parser.add_argument('--d-k', default=32, type=int,
                        help='Dimension of segment')
    parser.add_argument('--d-model', default=100, type=int,
                        help='Dimension of model')
    parser.add_argument('--d-embed', default=100, type=int,
                        help='Dimension of embedding')
    parser.add_argument('--d-inner', default=2048, type=int,
                        help='Dimension of inner (PL)')
    parser.add_argument('--label-smoothing', default=0.1, type=float,
                        help='label smoothing')
    parser.add_argument('--dr-in', default=0.4, type=float,
                        help='Intput dropout')
    parser.add_argument('--dr-hi', default=0.3, type=float,
                        help='Hidden dropout')
    parser.add_argument('--dr-fm', default=0.2, type=float,
                        help='Feature map dropout')
    parser.add_argument('--decay-until', default=1050, type=int,
                        help='decay learning rate until')
    parser.add_argument('--MR-method', default='gcn', type=str,
                        help='mlp, gcn or cnn ')

    args = parser.parse_args()
    main(args)
