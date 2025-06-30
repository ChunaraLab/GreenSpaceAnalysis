import ever as er

import torch
import numpy as np
import os
from data.loveda import COLOR_MAP
from tqdm import tqdm
import random
from os.path import exists
from module.tta import tta, Scale
from module.viz import VisualizeSegmm

er.registry.register_all()


def evaluate_cls_fn(self, test_dataloader, config=None):
    self.model.eval()
    classes = self.model.config.classes if self.model.config.classes != 1 else 2
    print("********* Number of classes is: " + str(classes) + " *********")
    
    metric_op = er.metric.PixelMetric(classes, logdir=self._model_dir, logger=self.logger)

    vis_dir = os.path.join(self._model_dir, 'vis-{}'.format(self.checkpoint.global_step))

    palette = np.array(list(COLOR_MAP.values())).reshape(-1).tolist()
    viz_op = VisualizeSegmm(vis_dir, palette)

    with torch.no_grad():
        for img, gt in tqdm(test_dataloader):
            img = img.to(torch.device('cuda'))
            y_true = gt['cls']
            y_true = y_true.cpu()
            if config.get('tta', False):
                pred = tta(self.model, img, tta_config=[
                    Scale(scale_factor=0.5),
                    Scale(scale_factor=0.75),
                    Scale(scale_factor=1.0),
                    Scale(scale_factor=1.25),
                    Scale(scale_factor=1.5),
                    Scale(scale_factor=1.75),
                ])
            else:
                pred = self.model(img)
            pred = pred.argmax(dim=1).cpu()

            valid_inds = y_true != -1
            metric_op.forward(y_true[valid_inds], pred[valid_inds])

            # for clsmap, imname in zip(pred, gt['fname']):
            #     viz_op(clsmap.cpu().numpy().astype(np.uint8), imname.replace('tif', 'png'))
    tb, iou_per_class, precision_per_class, recall_per_class = metric_op.summary_all()
    
    torch.save(self.model.state_dict(), 'log/normal_baseline/checkpoint'+str(iou_per_class.mean())+'.pth')
    
    if exists('log/normal_baseline/ious.npy'):
        ious = np.load('log/normal_baseline/ious.npy')
        precisions = np.load('log/normal_baseline/precisions.npy')
        recalls = np.load('log/normal_baseline/recalls.npy')
        
        iou_per_class = list(ious) + [iou_per_class[0], iou_per_class[1]]
        precision_per_class = list(precisions) + [precision_per_class[0], precision_per_class[1]]
        recall_per_class = list(recalls) + [recall_per_class[0], recall_per_class[1]]
        
        # iou_per_class = list(ious) + [iou_per_class[0], iou_per_class[1], iou_per_class[2]]
        # precision_per_class = list(precisions) + [precision_per_class[0], precision_per_class[1], precision_per_class[2]]
        # recall_per_class = list(recalls) + [recall_per_class[0], recall_per_class[1], recall_per_class[2]]
    
    else:
        iou_per_class = [iou_per_class[0], iou_per_class[1]]
        precision_per_class = [precision_per_class[0], precision_per_class[1]]
        recall_per_class = [recall_per_class[0], recall_per_class[1]]
    
    print(iou_per_class)
    
    torch.cuda.empty_cache()


def register_evaluate_fn(launcher):
    launcher.override_evaluate(evaluate_cls_fn)



def seed_torch(seed=2333):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed) # if you are using multi-GPU.
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.enabled = False



if __name__ == '__main__':
    seed_torch(2333)
    trainer = er.api.trainer.get_trainer('base')()
    trainer.run(after_construct_launcher_callbacks=[register_evaluate_fn])

