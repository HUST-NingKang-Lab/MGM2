"""Patched community encoder and token/CLS reconstruction in an isolated module."""
import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
from torch.utils.checkpoint import checkpoint
CONFIG = {'hidden_size': 224, 'layers': 6, 'heads': 4, 'intermediate_size': 896, 'embedding_dim': 1536, 'qwen_dim': 4096, 'dropout': 0.1, 'attention_dropout': 0.1, 'layer_norm_eps': 1e-12, 'regression_weight': 0.3, 'qwen_weight': 0.1, 'temperature': 0.07}
VARIANTS = {"S0": (1., 0.), "S1": (0., 1.)}
from ._backbone import BaseModel
from ._heads import AbundanceBinHead, RegressionHead, Qwen3AlignmentHead


class CheckpointEncoder(nn.Module):
    def __init__(self, encoder):
        super().__init__();self.layers=encoder.layers;self.norm=encoder.norm

    def forward(self, hidden, src_key_padding_mask=None):
        for layer in self.layers:
            if self.training and torch.is_grad_enabled():
                hidden=checkpoint(layer,hidden,src_key_padding_mask=src_key_padding_mask,use_reentrant=False)
            else:hidden=layer(hidden,src_key_padding_mask=src_key_padding_mask)
        return self.norm(hidden) if self.norm is not None else hidden


def build_backbone(config=CONFIG, legacy=False):
    model=BaseModel(embedding_dim=config['embedding_dim'],hidden_size=config['hidden_size'],
        num_hidden_layers=config['layers'],num_attention_heads=config['heads'],
        intermediate_size=config['intermediate_size'],hidden_dropout_prob=config['dropout'],
        attention_probs_dropout_prob=config['attention_dropout'],layer_norm_eps=config['layer_norm_eps'])
    if not legacy:
        for layer in model.encoder.layers:
            nn.init.xavier_uniform_(layer.self_attn.in_proj_weight)
            nn.init.zeros_(layer.self_attn.in_proj_bias)
            layer.self_attn.dropout=config['attention_dropout']
            layer.norm1.eps=layer.norm2.eps=config['layer_norm_eps']
        model.encoder=CheckpointEncoder(model.encoder)
    return model


class CLSDecoder(nn.Module):
    def __init__(self, config):
        super().__init__();d=config['hidden_size']
        self.identity=nn.Linear(config['embedding_dim'],d)
        self.fusion=nn.Sequential(nn.Linear(2*d,d),nn.GELU(),nn.LayerNorm(d),nn.Dropout(config['dropout']))
        self.bin_head=nn.Linear(d,5);self.regression_head=nn.Linear(d,1)

    def forward(self, cls, identity, sample_index):
        hidden=self.fusion(torch.cat([cls[sample_index],self.identity(identity)],dim=-1))
        return self.bin_head(hidden),self.regression_head(hidden).squeeze(-1)


class PilotModel(nn.Module):
    def __init__(self, config=CONFIG):
        super().__init__();self.backbone=build_backbone(config)
        d=config['hidden_size'];self.token_bin=AbundanceBinHead(d,5,config['dropout'])
        self.token_regression=RegressionHead(d,config['dropout'])
        self.cls_decoder=CLSDecoder(config)
        self.qwen_head=Qwen3AlignmentHead(d,config['qwen_dim'],config['dropout'])

    def forward(self, batch, variant):
        inputs={k:batch[k] for k in ['input_ids','embeddings','abundance_values','attention_mask']}
        hidden=self.backbone(**inputs)['last_hidden_state'];cls=hidden[:,0]
        selected=batch['selected'];sample_index=selected.nonzero(as_tuple=True)[0]
        result=dict(hidden=hidden,cls=cls,qwen=self.qwen_head(cls))
        wt,wc=VARIANTS[variant]
        if wt:
            tokens=hidden[selected];result['token']=(self.token_bin(tokens),self.token_regression(tokens))
        if wc:
            # Identity is the static upstream vector, never contextual token state
            # or abundance. This is the CLS information bottleneck.
            result['cls_reconstruction']=self.cls_decoder(cls,batch['embeddings'][selected],sample_index)
        return result


def corrupt_abundance(original, valid, seed, mask_prob=.15, ratios=(.8,.1,.1)):
    if not 0<=mask_prob<=1 or len(ratios)!=3 or not np.isclose(sum(ratios),1):raise ValueError('Invalid masking config')
    generator=torch.Generator(device=original.device).manual_seed(int(seed))
    counts=valid.sum(1)
    k=(counts.float()*mask_prob).long()
    if mask_prob>0:k=torch.where(counts>0,k.clamp_min(1),k)
    scores=torch.rand(original.shape,device=original.device,generator=generator).masked_fill(~valid,2.)
    ranks=scores.argsort(dim=1).argsort(dim=1);selected=(ranks<k[:,None])&valid
    branches=torch.rand(original.shape,device=original.device,generator=generator)
    positions=torch.arange(original.shape[1],device=original.device)[None,:].expand_as(original)
    choices=positions.masked_fill(~valid,original.shape[1]).sort(dim=1).values
    pick=(torch.rand(original.shape,device=original.device,generator=generator)*counts[:,None]).long()
    donor=choices.gather(1,pick).clamp_max(original.shape[1]-1)
    output=torch.where(selected&(branches<ratios[0]),-3.,original)
    random_branch=selected&(branches>=ratios[0])&(branches<ratios[0]+ratios[1])
    output=torch.where(random_branch,original.gather(1,donor),output)
    return output,selected


def alignment_loss(projection, target, groups, temperature=.07):
    valid=groups>=0;projection=projection[valid].float();target=target[valid].float();groups=groups[valid]
    n=len(groups)
    if n<2:return projection.sum()*0,n
    with torch.autocast(device_type=projection.device.type,enabled=False):
        logits=F.normalize(projection,dim=-1)@F.normalize(target,dim=-1).T/temperature
    positives=groups[:,None]==groups[None,:]
    positive_logits=logits.masked_fill(~positives,float('-inf'))
    loss=.5*((logits.logsumexp(1)-positive_logits.logsumexp(1)).mean()+
              (logits.logsumexp(0)-positive_logits.logsumexp(0)).mean())
    return loss,n


def reconstruction_sums(logits,prediction,bins,target):
    logits=logits.float();prediction=prediction.float();target=target.float()
    n=bins.numel();reg=bins>=3;nr=reg.sum()
    if n:
        ce=F.cross_entropy(logits,bins.long(),reduction='none')
        focal=(.25*(1-torch.exp(-ce)).square()*ce).sum()
    else:focal=logits.sum()*0
    residual=prediction[reg]-target[reg]
    mse=residual.square().sum();mae=residual.abs().sum()
    loss=focal/max(n,1)+CONFIG['regression_weight']*mse/nr.clamp_min(1)
    # Sufficient statistics, not means of batch means/correlations.
    r=prediction[reg];y=target[reg]
    stats=torch.stack([focal.detach(),torch.as_tensor(n,device=logits.device),
        ((logits.argmax(-1)==bins).sum()).float(),mse.detach(),mae.detach(),nr.float(),
        r.detach().sum(),y.sum(),r.detach().square().sum(),y.square().sum(),(r.detach()*y).sum()]).double()
    return loss,stats


def objective(outputs,batch,variant):
    selected=batch['selected'];bins=batch['bins'][selected];target=batch['target_abundance'][selected]
    loss=outputs['cls'].sum()*0;stats={}
    for key,weight in zip(['token','cls_reconstruction'],VARIANTS[variant]):
        if weight:
            term,stats[key]=reconstruction_sums(*outputs[key],bins,target);loss=loss+weight*term
    align,n=alignment_loss(outputs['qwen'],batch['qwen'],batch['text_groups'],CONFIG['temperature'])
    loss=loss+CONFIG['qwen_weight']*align
    stats['qwen']=torch.stack([align.detach().double()*n,torch.tensor(n,device=align.device,dtype=torch.float64)])
    return loss,stats


def summarize_statistics(sums,variant):
    result={};total=0.
    for key,weight in zip(['token','cls_reconstruction'],VARIANTS[variant]):
        if not weight:continue
        s=np.asarray(sums[key]);focal=s[0]/max(s[1],1);mse=s[3]/max(s[5],1)
        denom=np.sqrt(max(s[5]*s[8]-s[6]**2,0)*max(s[5]*s[9]-s[7]**2,0))
        corr=(s[5]*s[10]-s[6]*s[7])/denom if denom>0 else None
        result[key]=dict(focal=focal,accuracy=s[2]/max(s[1],1),mse=mse,mae=s[4]/max(s[5],1),
            correlation=corr,masked_tokens=int(s[1]),regression_tokens=int(s[5]))
        total+=weight*(focal+CONFIG['regression_weight']*mse)
    q=np.asarray(sums['qwen']);result['qwen_loss']=float(q[0]/max(q[1],1));result['qwen_anchors']=int(q[1])
    result['total_loss']=float(total+CONFIG['qwen_weight']*result['qwen_loss'])
    return result
