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

    def voxel_wise_loss_function(self, _x, _target, _scale=1e3, _dims=[]):
        return _scale * torch.mean(torch.square(_x[:, :4] - _target), _dims)

    def begin(self):
        # create optimizer and lr scheduler
        optimizer = torch.optim.Adam(
            self.model.parameters(),
            lr=self.args.start_lr, 
            weight_decay=0.5
        )
        lr_scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer=optimizer,
            factor=self.args.factor_sched,
            patience=self.args.patience_sched,
            min_lr=self.args.end_lr,
        )

        # create seed and target tensors
        seed_ten = utils.load_vox_as_tensor(self.args.seed).to('cpu')
        target_ten = utils.load_vox_as_tensor(self.args.target).to('cpu')
        import torch.nn.functional as func
        pad = utils.DEFAULT_PAD
        target_ten = func.pad(target_ten, (pad, pad, pad, pad, pad, pad), 'constant')
        target_size = target_ten.shape[-1]

        # resize seed_ten s.t. it is the same size as target_tensor
        seed_size = seed_ten.shape[-1]
        seed_pad = (target_size-seed_size)//2
        extra_pad = 0
        if seed_size+(seed_pad*2) < target_size: extra_pad = 1
        assert target_size == seed_size+(seed_pad*2)+extra_pad
        seed_ten = func.pad(seed_ten, (seed_pad+extra_pad, seed_pad, seed_pad+extra_pad, seed_pad, seed_pad+extra_pad, seed_pad), 'constant')
        target_ten_bs = target_ten.clone().repeat(self.args.batch_size, 1, 1, 1, 1)

        # augment tensors with hidden channels
        cur_channels = seed_ten.shape[1]
        add_channels = self.args.channels-cur_channels
        augment_shape = [1, add_channels, target_size, target_size, target_size]
        seed_ten = torch.cat([seed_ten, torch.zeros(augment_shape)], dim=1)
        target_ten = torch.cat([target_ten, torch.zeros(augment_shape)], dim=1)

        # set seed hidden channels to 1 if cell is alive (alpha == 1)
        for x in range(target_size):
            for y in range(target_size):
                for z in range(target_size):
                    if seed_ten[0, 3, x, y, z] == 1:
                        seed_ten[0, 3:self.args.channels, x, y, z] = 1

        # create pool tensor
        from scripts.nca.perception import orientation_channels
        isotype = orientation_channels(self.model.perception)
        pool = utils.generate_pool(self.args, seed_ten, isotype).to('cpu')
        
        utils.log(f'{PROGRAM} seed.shape: {Fore.WHITE}{list(seed_ten.shape)}{Style.RESET_ALL}')
        utils.log(f'{PROGRAM} target.shape: {Fore.WHITE}{list(target_ten.shape)}{Style.RESET_ALL}')
        utils.log(f'{PROGRAM} pool.shape:{Fore.WHITE}{list(pool.shape)}{Style.RESET_ALL}')
        utils.log(f'{PROGRAM} starting training w/ {Fore.GREEN}{self.args.epochs}{Style.RESET_ALL} epochs...')

        loss_log = []
        min_avg_loss = 1e100
        best_model_path = None
        train_start = datetime.datetime.now()
        for epoch in range(self.args.epochs+1):
            with torch.no_grad():
                # * sample batch from pool
                batch_idxs = np.random.choice(self.args.pool_size, self.args.batch_size, replace=False)
                x = pool[batch_idxs]
                
                # * re-order batch based on loss
                loss_ranks = torch.argsort(self.voxel_wise_loss_function(x, target_ten_bs, _dims=[-1, -2, -3, -4]), descending=True)
                x = x[loss_ranks]
                
                # * re-add seed into batch
                x[:1] = seed_ten.clone()
                # * randomize last channel
                if isotype == 1:
                    x[:1, -1:] = torch.rand(target_size, target_size, target_size)*np.pi*2.0
                elif isotype == 3:
                    x[:1, -1:] = torch.rand(target_size, target_size, target_size)*np.pi*2.0
                    x[:1, -2:-1] = torch.rand(target_size, target_size, target_size)*np.pi*2.0
                    x[:1, -3:-2] = torch.rand(target_size, target_size, target_size)*np.pi*2.0
            
                # * damage lowest loss in batch
                if epoch % self.args.damage_rate == 0:
                    mask = torch.tensor(utils.half_volume_mask(target_size, 'rand'))
                    # * apply mask
                    x[-self.args.damage_num:] *= mask
                    # * randomize angles for steerable models
                    if isotype == 1:
                        inv_mask = ~mask
                        x[-self.args.damage_num:, -1:] += torch.rand(target_size, target_size, target_size)*np.pi*2.0*inv_mask
                    elif isotype == 3:
                        inv_mask = ~mask
                        x[-self.args.damage_num:, -1:] += torch.rand(target_size, target_size, target_size)*np.pi*2.0*inv_mask
                        x[-self.args.damage_num:, -2:-1] += torch.rand(target_size, target_size, target_size)*np.pi*2.0*inv_mask
                        x[-self.args.damage_num:, -3:-2] += torch.rand(target_size, target_size, target_size)*np.pi*2.0*inv_mask

            # * different loss values
            overflow_loss = 0.0
            diff_loss = 0.0
            target_loss = 0.0
            
            # * forward pass
            x = x.to('cuda')
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
            x = x.to('cpu')
            target_loss += self.voxel_wise_loss_function(x, target_ten_bs)
            target_loss /= 2.0
            diff_loss *= 10.0
            loss = target_loss + overflow_loss + diff_loss
    
            # backwards pass
            with torch.no_grad():
                loss.backward()
                optimizer.step()
                optimizer.zero_grad()
                lr_scheduler.step(loss)
                _loss = loss.item()

            # with torch.no_grad():
            #     loss.backward()
            #     # * normalize gradients 
            #     for p in self.model.parameters():
            #         p.grad /= (p.grad.norm()+1e-5)

            #     torch.nn.utils.clip_grad_norm_(self.model.parameters(), 5) # maybe? : 
                optimizer.step()
                optimizer.zero_grad()
                # self.sched.step()

            # * re-add batch to pool
            pool[batch_idxs] = x.clone()
            # * correctly add to loss log
            _loss = loss.item()
            if torch.isnan(loss) or torch.isinf(loss) or torch.isneginf(loss):
                pass
            else:
                loss_log.append(_loss)
            avg_loss = sum(loss_log[-self.args.info_rate:])/float(self.args.info_rate)
                                
            # * detect invalid loss values :(
            if torch.isnan(loss) or torch.isinf(loss) or torch.isneginf(loss):
                utils.log(f'{PROGRAM} {Fore.RED}error!{Style.RESET_ALL} detected invalid loss value: {loss}')
                utils.log(f'{PROGRAM} overflow loss: {overflow_loss}, diff loss: {diff_loss}, target loss: {target_loss}')
                raise ValueError
            
            # * print info
            if epoch % self.args.info_rate == 0 and epoch!= 0:
                secs = (datetime.datetime.now()-train_start).seconds
                time = str(datetime.timedelta(seconds=secs))
                iter_per_sec = float(epoch)/float(secs)
                est_time_sec = int((self.args.epochs-epoch)*(1/iter_per_sec))
                est = str(datetime.timedelta(seconds=est_time_sec))
                lr = np.round(lr_scheduler.get_last_lr()[0], 8)
                utils.log(f'{PROGRAM} [{Fore.CYAN}{epoch}{Style.RESET_ALL}/{self.args.epochs}]\t {Fore.CYAN}{np.round(iter_per_sec, 3)}{Style.RESET_ALL}it/s\t time: {Fore.CYAN}{time}{Style.RESET_ALL}~{est}\t loss: {Fore.CYAN}{np.round(avg_loss, 3)}{Style.RESET_ALL}>{np.round(np.min(loss_log), 3)}\t lr: {lr}')
            
            # save model if minimun average loss detected
            if avg_loss < min_avg_loss and epoch >= self.args.info_rate:
                min_avg_loss = avg_loss
                if best_model_path is not None:
                    os.remove(best_model_path)
                best_model_path = f'models/{self.args.model_dir}/best@{epoch}.pt'
                self.model.save(f'best@{epoch}')
                utils.log(f'{PROGRAM} detected minimum average loss during training: {Fore.GREEN}{np.round(min_avg_loss, 3)}{Style.RESET_ALL} -- saving model to: {Fore.WHITE}{best_model_path}{Style.RESET_ALL}')
                
        
        # * save loss plot
        pl.plot(loss_log, '.', alpha=0.1)
        pl.yscale('log')
        pl.ylim(np.min(loss_log), loss_log[0])
        pl.savefig(f'models/{self.args.model_dir}/train_loss_plot.png')
    
        # * calculate elapsed time
        secs = (datetime.datetime.now()-train_start).seconds
        elapsed_time = str(datetime.timedelta(seconds=secs))
        utils.log(f'{PROGRAM} elapsed time: {Fore.WHITE}{elapsed_time}{Style.RESET_ALL}')

        # * save final model
        self.model.save(f'final@{epoch}')
        utils.log(f'{PROGRAM} {Fore.GREEN}training complete{Style.RESET_ALL} -- saving final model to: {Fore.WHITE}{self.args.model_dir}/{self.args.name}/final@{epoch}.pt{Style.RESET_ALL}')
