from colorama import init, Style, Fore
import scripts.nca.utils as utils
import matplotlib.pylab as pl
import numpy as np
import datetime
import torch
import os

init()
PROGRAM = f'{Style.DIM}[{os.path.basename(__file__)}]{Style.RESET_ALL}'

# template nca trainer - build new trainers from this
class _base_nca_trainer_():
    def __init__(
        self,
        _model: torch.nn.Module,
    ):
        self.model = _model
        self.args = _model.args

    def begin(self):
        raise NotImplementedError
        # this method must be implemented by sub-classes!

class thesis_nca_trainer(_base_nca_trainer_):
    def __init__(
        self,
        _model: torch.nn.Module,
    ):
        super(thesis_nca_trainer, self).__init__(_model)

    def voxel_wise_loss_function(_x, _target, _scale=1e3, _dims=[]):
        return _scale * torch.mean(torch.square(_x[:, :4] - _target), _dims)

    def begin(self):
        start = datetime.datetime.now()

        # create optimizer and lr scheduler
        optimizer = torch.optim.Adam(
            self.model.parameters(),
            lr=self.args.start_lr, 
            weight_decay=0.5
        )
        lr_scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer=optimizer,
            factor=self.args.factor_sch,
            patience=self.args.patience_sch,
            min_lr=self.args.end_lr,
        )

        # create seed and target tensors
        seed_ten = utils.load_vox_as_tensor(self.args.seed).to('cuda')
        target_ten = utils.load_vox_as_tensor(self.args.target).to('cuda')
        import torch.nn.functional as func
        pad = utils.DEFAULT_PAD
        target_ten = func.pad(target_ten, (pad, pad, pad, pad, pad, pad), 'constant')
        size = target_ten.shape[-1]

        # resize seed_ten s.t. it is the same size as target_tensor
        seed_pad = (size-seed_ten.shape[-1])//2
        seed_ten = func.pad(target_ten, (seed_pad, seed_pad, seed_pad, seed_pad, seed_pad, seed_pad), 'constant')
        target_ten_bs = target_ten.clone().repeat(self.args.batch_size, 1, 1, 1, 1)
        utils.log(f'{PROGRAM} seed.shape: {seed_ten.shape}')
        utils.log(f'{PROGRAM} target.shape: {target_ten.shape}')

        # create pool tensor
        from scripts.nca.perception import orientation_channels
        isotype = orientation_channels(self.model.perception)
        pool = utils.generate_pool(self.args, seed_ten, isotype).to('cuda')

        utils.log(f'beginning training w/ {self.epochs} epochs...')

        loss_log = []
        prev_lr = -np.inf
        for i in range(self.epochs+1):
            with torch.no_grad():
                # * sample batch from pool
                batch_idxs = np.random.choice(self.args.pool_size, self.args.batch_size, replace=False)
                x = pool[batch_idxs]
                
                # * re-order batch based on loss
                loss_ranks = torch.argsort(self.voxel_wise_loss_function(x, target_ten_bs, _dims=[-1, -2, -3, -4]), descending=True)
                x = x[loss_ranks]
                
                # * re-add seed into batch
                x[:1] = self.seed
                # * randomize last channel
                if isotype == 1:
                    x[:1, -1:] = torch.rand(size, size, size)*np.pi*2.0
                elif isotype == 3:
                    x[:1, -1:] = torch.rand(size, size, size)*np.pi*2.0
                    x[:1, -2:-1] = torch.rand(size, size, size)*np.pi*2.0
                    x[:1, -3:-2] = torch.rand(size, size, size)*np.pi*2.0
            
                # * damage lowest loss in batch
                if i % self.args.damage_rate == 0:
                    mask = torch.tensor(utils.half_volume_mask(size, 'rand'))
                    # * apply mask
                    x[-self.args:] *= mask
                    # * randomize angles for steerable models
                    if self.isotype == 1:
                        inv_mask = ~mask
                        x[-self.args.damage_num:, -1:] += torch.rand(size, size, size)*np.pi*2.0*inv_mask
                    elif self.isotype == 3:
                        inv_mask = ~mask
                        x[-self.args.damage_num:, -1:] += torch.rand(size, size, size)*np.pi*2.0*inv_mask
                        x[-self.args.damage_num:, -2:-1] += torch.rand(size, size, size)*np.pi*2.0*inv_mask
                        x[-self.args.damage_num:, -3:-2] += torch.rand(size, size, size)*np.pi*2.0*inv_mask

            # * different loss values
            overflow_loss = 0.0
            diff_loss = 0.0
            target_loss = 0.0
            
            # * forward pass
            num_steps = np.random.randint(64, 96)
            for _ in range(num_steps):
                prev_x = x
                x = self.model(x)
                diff_loss += (x - prev_x).abs().mean()
                if isotype == 1:
                    overflow_loss += (x - x.clamp(-2.0, 2.0))[:, :self.args.channels-1].square().sum()
                elif isotype == 3:
                    overflow_loss += (x - x.clamp(-2.0, 2.0))[:, :self.args.channels-3].square().sum()
                else:
                    overflow_loss += (x - x.clamp(-2.0, 2.0))[:, :self.args.channels].square().sum()
            
            # * calculate losses
            target_loss += self.voxel_wise_loss_function(x, target_ten_bs)
            target_loss /= 2.0
            diff_loss *= 10.0
            loss = target_loss + overflow_loss + diff_loss
            
            # * backward pass
            with torch.no_grad():
                loss.backward()
                # * normalize gradients 
                for p in self.model.parameters():
                    p.grad /= (p.grad.norm()+1e-5)

                torch.nn.utils.clip_grad_norm_(self.model.parameters(), 5) # maybe? : 
                optimizer.step()
                optimizer.zero_grad()
                self.sched.step()
                # * re-add batch to pool
                pool[batch_idxs] = x
                # * correctly add to loss log
                _loss = loss.item()
                if torch.isnan(loss) or torch.isinf(loss) or torch.isneginf(loss):
                    pass
                else:
                    loss_log.append(_loss)
                                    
                # * detect invalid loss values :(
                if torch.isnan(loss) or torch.isinf(loss) or torch.isneginf(loss):
                    logprintDDP(f'models/{name}/{logf}', f'detected invalid loss value: {loss}', self.gpu_id)
                    logprintDDP(f'models/{name}/{logf}', f'overflow loss: {overflow_loss}, diff loss: {diff_loss}, target loss: {target_loss}', self.gpu_id)
                    raise ValueError
                
                # * print info
                if i % info == 0 and i!= 0:
                    secs = (datetime.datetime.now()-start).seconds
                    time = str(datetime.timedelta(seconds=secs))
                    iter_per_sec = float(i)/float(secs)
                    est_time_sec = int((epochs-i)*(1/iter_per_sec))
                    est = str(datetime.timedelta(seconds=est_time_sec))
                    avg = sum(loss_log[-info:])/float(info)
                    lr = np.round(self.sched.get_last_lr()[0], 8)
                    step = '▲'
                    if prev_lr > lr:
                        step = '▼'
                    prev_lr = lr
                    logprintDDP(f'models/{name}/{logf}', f'[{i}/{epochs+1}]\t {np.round(iter_per_sec, 3)}it/s\t time: {time}~{est}\t loss: {np.round(avg, 3)}>{np.round(np.min(loss_log), 3)}\t lr: {lr} {step}', self.gpu_id)
                
                # * save checkpoint
                if i % save == 0 and i != 0 and self.gpu_id == 0:
                    self.model.save('models/checkpoints', name+'_cp'+str(i), self.nca_params)
                    logprintDDP(f'models/{name}/{logf}', f'model [{name}] saved to checkpoints...', self.gpu_id)
        
        # * save loss plot
        if self.gpu_id == 0:
            pl.plot(loss_log, '.', alpha=0.1)
            pl.yscale('log')
            pl.ylim(np.min(loss_log), loss_log[0])
            pl.savefig(f'models/{name}/{name}_loss_plot.png')
                        
            # * save final model
            self.model.save('models', name+'_final', self.nca_params)
        
            # * calculate elapsed time
            secs = (datetime.datetime.now()-start).seconds
            elapsed_time = str(datetime.timedelta(seconds=secs))
            logprintDDP(f'models/{name}/{logf}', f'elapsed time: {elapsed_time}', self.gpu_id)
            logprintDDP(f'models/{name}/{logf}', '****************', self.gpu_id)
