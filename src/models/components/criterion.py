import torch
import numpy as np
import torch.nn.functional as F
from types import SimpleNamespace

class KDCriterion:
    def __init__(self, **kwargs) -> None:
        args = SimpleNamespace(**kwargs)
        self.args = args
        self.criterion_aligned_img_kd = args.img_criterion
        self.criterion_nlp_kd = args.nlp_criterion
        self.temperature = args.temperature #2
        logit_scale = torch.nn.Parameter(torch.ones([]) * np.log(1 / 0.07), requires_grad = False)
        self.logit_scale = logit_scale.exp()

    def __call__(self, inputs):
        hidden_features, out, clip_img_features, clip_nlp_features, aligned_img, aligned_nlp = inputs
        img_loss = self.criterion_aligned_img_kd(hidden_features, aligned_img)

        logit_scale = self.logit_scale.to(hidden_features.device)
        student_nlp_logits = logit_scale * hidden_features @ aligned_nlp.T / self.temperature
        teacher_nlp_logits = logit_scale * clip_img_features @ clip_nlp_features.T / self.temperature
        kd_loss = self.criterion_nlp_kd(F.log_softmax(student_nlp_logits, dim=1),
                             F.softmax(teacher_nlp_logits, dim=1)) * (self.temperature * self.temperature)
        kd_loss = kd_loss * self.args.class_num / 2
        
        return img_loss, kd_loss
