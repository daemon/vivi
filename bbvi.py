from collections import defaultdict, namedtuple

from tqdm import tqdm
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.distributions.one_hot_categorical as cat
import torch.distributions.relaxed_categorical as rcat


SurrogateConfig = namedtuple("SurrogateConfig", "width, depth, activation, n_classes")

class VIVIEstimator(nn.Module):

    def __init__(self, embedding, target_model, language_model, surrogate_config=SurrogateConfig(200, 2, nn.Tanh, 3), 
            length=15, batch_size=64):
        super().__init__()
        self.wordvec_params = nn.Parameter(torch.empty(length, embedding.embedding_dim).uniform_() - 0.5)
        self.surrogate_config = surrogate_config
        self.embedding = embedding
        self.logits = nn.Parameter(torch.empty(length, embedding.num_embeddings).uniform_() - 0.5)
        self.target_model = target_model
        self.language_model = language_model
        self.batch_size = batch_size
        self.length = length
        self.surrogate = self._build_surrogate()

    def sample(self):
        logits = F.linear(self.wordvec_params, self.embedding.weight)
        c_dist = cat.OneHotCategorical(logits=logits)
        return c_dist.sample()

    def list_params(self):
        return [self.wordvec_params] + list(self.surrogate.parameters())

    def _build_surrogate(self):
        depth = self.surrogate_config.depth
        width = self.surrogate_config.width
        n_classes = self.surrogate_config.n_classes
        activation = self.surrogate_config.activation
        layers = [nn.Linear(self.length * self.embedding.num_embeddings, width), activation()]
        for _ in range(depth - 1):
            layers += [nn.Linear(width, width), activation()]
        layers += [nn.Linear(width, 1)]
        return nn.Sequential(*layers)

    def forward(self, target_class, eps=1E-5, reinforce=False):
        input_dict = defaultdict(list)
        logits = F.linear(self.wordvec_params, self.embedding.weight)
        c_dist = cat.OneHotCategorical(logits=logits)
        rc_dist = rcat.RelaxedOneHotCategorical(2 / 3, logits=logits)
        probs = F.softmax(logits, 1)
        logp_btheta = []
        indices_lst = []
        z_lst = []
        z_tilde_lst = []
        for _ in range(self.batch_size):
            if reinforce:
                b = c_dist.sample()
                indices_lst.append(b.max(1)[1])
                logp_btheta.append(c_dist.log_prob(b).sum())
                continue
            z = rc_dist.rsample()
            indices = z.max(1)[1]
            b = torch.zeros_like(z).to(z.device)
            b[torch.arange(0, b.size(0)), indices] = 1
            logp_btheta.append(c_dist.log_prob(b).sum())
            u = torch.empty(*z.size()).uniform_().to(z.device).clamp_(eps, 1 - eps)
            u.requires_grad = True
            vb = u[b.byte()].unsqueeze(-1).expand_as(probs)
            z_tilde = u.log().mul(-1).log().mul(-1).mul(b) + \
                u.log().mul(-1).div(probs).sub(vb.log()).log().mul(-1).mul(1 - b)
            indices_lst.append(indices)
            z_lst.append(z)
            z_tilde_lst.append(z_tilde)

        if not reinforce:
            z = torch.stack(z_lst)
            z_tilde = torch.stack(z_tilde_lst)
            c1 = self.surrogate(z.view(z.size(0), -1)).squeeze(-1)
            c2 = self.surrogate(z_tilde.view(z_tilde.size(0), -1)).squeeze(-1)
        else:
            c1 = c2 = 0

        indices = torch.stack(indices_lst)
        lp_tm = F.log_softmax(self.target_model(indices), 1)
        lp_tm = lp_tm[:, target_class]
        lp_lm = self.language_model(indices).sum(1)
        entropy = (c_dist.entropy() * self.batch_size).mean()

        f_b = -lp_tm - lp_lm - entropy
        loss = (f_b - c2).detach() * torch.stack(logp_btheta) + c1 - c2
        loss = loss.mean()
        if not reinforce:
            torch.autograd.backward(loss, create_graph=True, retain_graph=True)
            loss_grad = torch.autograd.grad([loss], [self.logits], create_graph=True, retain_graph=True)[0]
            torch.autograd.backward((loss_grad ** 2).mean(), create_graph=True, retain_graph=True)
        else:
            loss.backward()
        return loss.item()


class PhonyTargetModel(nn.Module):

    def __init__(self):
        super().__init__()

    def forward(self, x):
        y = torch.zeros(64, 3).float().to(x.device)
        y[:, 1] = (x == 1).float().sum() / 5
        return y


class PhonyLanguageModel(nn.Module):

    def __init__(self):
        super().__init__()

    def forward(self, x):
        lm_x = x.clone().detach().float() * 0
        # lm_x[x == 0] = 0.5
        # lm_x[x == 1] = 0.3
        # lm_x[x == 2] = 0.2
        return F.log_softmax(lm_x, 1)


def phony_test_main():
    import torch.optim as optim
    import numpy as np
    sums = []
    for _ in range(5):
        embedding = nn.Embedding.from_pretrained(torch.zeros(300, 1000).uniform_() * 10)
        target_model = PhonyTargetModel()
        language_model = PhonyLanguageModel()
        estimator = VIVIEstimator(embedding, target_model, language_model)
        optimizer = optim.Adam(estimator.list_params(), lr=0.1)#, weight_decay=1E-3)
        pbar = tqdm(range(1000))
        for idx in pbar:
            optimizer.zero_grad()
            if idx % 200 == 0:
                optimizer.param_groups[0]["lr"] /= 10
            loss = estimator(1, reinforce=True)
            optimizer.step()
            pbar.set_postfix(dict(loss=f"{loss:.3}"))
            print(estimator.sample()[:, 1].sum().item())
        pbar.close()
        print(estimator.sample()[:, 1].sum().item())
    print(np.mean(sums))

if __name__ == "__main__":
    phony_test_main()