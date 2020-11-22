#coding:utf8
from itertools import chain
import visdom
import torch as t
import time
import torchvision as tv
import numpy as np
import torch

IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224,  0.225]

def gram_matrix(y):
    '''
    输入 b,c,h,w
    输出 b,c,c
    '''
    (b, ch, h, w) = y.size()
    features = y.view(b, ch, w * h)
    features_t = features.transpose(1, 2)
    gram = features.bmm(features_t) / (ch * h * w)
    return gram

def TemporalLoss( x, f_x1, cm):

    assert x.shape == f_x1.shape  ,"inputs are ain't same"
    x = x.reshape(1, -1)
    f_x1 = f_x1.reshape(1, -1)
    D = f_x1.shape[1]
    # print(x.shape,f_x1.shape)   #(1, 691200) (1, 691200)
    cm = torch.from_numpy(cm.reshape(-1)).float()
    cm = cm.cuda()
    # print(cm.shape) # (230400,)
    return ((1 / D) * torch.sum( cm * ((x-f_x1)**2)))

def PROCESS(d,x2):
    ma = torch.max(d)
    mi = torch.min(d)
    dn = (d - mi) / (ma - mi)
    # print(dn)
    b, c, w, h = x2.shape
    x1 = torch.zeros((b, c, w, h)).cuda()
    one = torch.ones((b, w, h)).cuda()
    aa = dn[:, 0, :, :] + one
    # print(aa)
    for i in range(c):
        x1[:, i, :, :] = x2[:, i, :, :] * aa
    return x1
class Visualizer():
    '''
    封装了visdom的基本操作，但是你仍然可以通过`self.vis.function`
    调用原生的visdom接口
    '''

    def __init__(self, env='default', **kwargs):
        import visdom
        self.vis = visdom.Visdom(env=env, **kwargs)
        
        # 画的第几个数，相当于横座标
        # 保存（’loss',23） 即loss的第23个点
        self.index = {} 
        self.log_text = ''
    def reinit(self,env='default',**kwargs):
        '''
        修改visdom的配置
        '''
        self.vis = visdom.Visdom(env=env,**kwargs)
        return self

    def plot_many(self, d):
        '''
        一次plot多个
        @params d: dict (name,value) i.e. ('loss',0.11)
        '''
        for k, v in d.items():
            self.plot(k, v)

    def img_many(self, d):
        for k, v in d.items():
            self.img(k, v)

    def plot(self, name, y):
        '''
        self.plot('loss',1.00)
        '''
        x = self.index.get(name, 0)
        self.vis.line(Y=np.array([y]), X=np.array([x]),
                      win=name,
                      opts=dict(title=name),
                      update=None if x == 0 else 'append'
                      )
        self.index[name] = x + 1

    def img(self, name, img_):
        '''
        self.img('input_img',t.Tensor(64,64))
        '''
         
        if len(img_.size())<3:
            img_ = img_.cpu().unsqueeze(0) 
        self.vis.image(img_.cpu(),
                       win=name,
                       opts=dict(title=name)
                       )



    def img_grid_many(self,d):
        for k, v in d.items():
            self.img_grid(k, v)

    def img_grid(self, name, input_3d):
        '''
        一个batch的图片转成一个网格图，i.e. input（36，64，64）
        会变成 6*6 的网格图，每个格子大小64*64
        '''
        self.img(name, tv.utils.make_grid(
            input_3d.cpu()[0].unsqueeze(1).clamp(max=1,min=0)))

    def log(self,info,win='log_text'):
        '''
        self.log({'loss':1,'lr':0.0001})
        '''

        self.log_text += ('[{time}] {info} <br>'.format(
                            time=time.strftime('%m%d_%H%M%S'),
                            info=info)) 
        self.vis.text(self.log_text,win='log_text')

    def __getattr__(self, name):
        return getattr(self.vis, name)


def get_style_data(path):
    '''
    加载风格图片，
    输入： path， 文件路径
    返回： 形状 1*c*h*w， 分布 -2~2
    '''
    style_transform = tv.transforms.Compose([
        tv.transforms.ToTensor(),
        tv.transforms.Normalize(mean = IMAGENET_MEAN,std = IMAGENET_STD),
    ])

    style_image = tv.datasets.folder.default_loader(path)
    style_tensor = style_transform(style_image)
    return style_tensor.unsqueeze(0)

def normalize_batch(batch):
    '''
    输入: b,ch,h,w  0~255
    输出: b,ch,h,w  -2~2
    '''
    mean = batch.data.new(IMAGENET_MEAN).view(1,-1,1,1)
    std = batch.data.new(IMAGENET_STD).view(1,-1,1,1)
    mean = t.autograd.Variable(mean.expand_as(batch.data))
    std = t.autograd.Variable(std.expand_as(batch.data))
    return (batch/255.0 - mean) / std

