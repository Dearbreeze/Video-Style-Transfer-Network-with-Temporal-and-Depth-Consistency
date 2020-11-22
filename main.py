#coding:utf8
from PIL import Image
import torchvision as tv
import torchnet as tnt
import torch as t
import torch
from torch.utils import data
import utils
from style_for_videos_test.PackedVGG import Vgg19
from torch.autograd import Variable
from torch.nn import functional as F
from torch import nn
import tqdm
import os
import warnings
import sys
import tqdm
import time
from DepthNet import HourGlass
import cv2
from torchvision import datasets, transforms
from dataset import get_loader
from transformer_net import TransformerNet
from utils import *
from opticalflow import opticalflow
from flow_warp import *

# warnings.filterwarnings("ignore")


mean =  [0.485, 0.456, 0.406]
std = [0.229, 0.224,  0.225]

class Config(object):
    batch_size = 1
    num_workers = 4 # 多线程加载数据
    use_gpu = True # 使用GPU
    
    style_path= './style2.jpg' # 风格图片存放路径
    lr = 1e-3 # 学习率

    env = '' # visdom env
    plot_every= 10 # 每10个batch可视化一次
    depth_path = '' ## DepthNet path
    video_path = '' ## test video path

    content_weight = 1 #  content_loss 的权重
    style_weight = 1e5 #  style_loss的权重
    temporal_weight = 200
    long_temporal_weight = 100
    depth_weight = 50
    # tv_weight = 1e-6
    sample_frames = 5
    model_path = '' # 预训练模型的路径


    epoch = 2
    data_path = '' ## train dataset path

    img_shape = (640, 360)
opt = Config()



def train(**kwargs):


    for k_,v_ in kwargs.items():
        setattr(opt, k_, v_)

    vis = utils.Visualizer(opt.env)
    
    # 数据加载
    transform = transforms.Compose([transforms.ToTensor(),
                                    transforms.Normalize((0.485, 0.456, 0.406),
                                                         (0.229, 0.224, 0.225))])
    loader = get_loader(batch_size=1, data_path=opt.data_path, img_shape=opt.img_shape, transform=transform)

    # 转换网络
    transformer = TransformerNet().cuda()
    # transformer.load_state_dict(t.load(opt.model_path, ))

    #if opt.model_path:
    #    transformer.load_state_dict(t.load(opt.model_path,map_location=lambda _s, _: _s))

    # 损失网络 Vgg16
    vgg = Vgg19().eval()
    depthnet = HourGlass().eval()
    depthnet.load_state_dict(t.load(opt.depth_path ))
    # print(vgg)
    # BASNET
    net = BASNet(3, 1).cuda()
    net.load_state_dict(torch.load('./basnet.pth'))
    net.eval()

    # 优化器
    optimizer = t.optim.Adam(transformer.parameters(),lr=opt.lr)

    # 获取风格图片的数据


    img = Image.open(opt.style_path)
    img = img.resize(opt.img_shape)
    img = transform(img).float()
    style = Variable(img, requires_grad=True).unsqueeze(0)
    vis.img('style',(style[0]*0.225+0.45).clamp(min=0,max=1))
    
    if opt.use_gpu:
        transformer.cuda()
        style = style.cuda()
        vgg.cuda()
        depthnet.cuda()

    # 风格图片的gram矩阵
    style_v = Variable(style,volatile=True)
    features_style = vgg(style_v)
    gram_style = [Variable(utils.gram_matrix(y.data)) for y in features_style]

    # 损失统计
    style_meter = tnt.meter.AverageValueMeter()
    content_meter = tnt.meter.AverageValueMeter()
    temporal_meter = tnt.meter.AverageValueMeter()
    long_temporal_meter = tnt.meter.AverageValueMeter()
    depth_meter = tnt.meter.AverageValueMeter()
    # tv_meter = tnt.meter.AverageValueMeter()
    kk = 0
    for count in range(opt.epoch):
        print('Training Start!!')
        content_meter.reset()
        style_meter.reset()
        temporal_meter.reset()
        long_temporal_meter.reset()
        depth_meter.reset()
        # tv_meter.reset()
        for step, frames in enumerate(loader):
            for i in tqdm.tqdm(range(1, len(frames))):
                kk += 1
                if (kk + 1) % 3000 == 0:
                    print('LR had changed')
                    for param in optimizer.param_groups:
                        param['lr'] = max(param['lr'] / 1.2, 1e-4)


                optimizer.zero_grad()
                x_t = frames[i].cuda()

                x_t1 = frames[i-1].cuda()

                h_xt = transformer(x_t)

                h_xt1 = transformer(x_t1)
                depth_x_t = depthnet(x_t)
                depth_x_t1 = depthnet(x_t1)
                depth_h_xt = depthnet(h_xt)
                depth_h_xt1 = depthnet(h_xt1)

                img1 = h_xt1.data.cpu().squeeze(0).numpy().transpose(1,2,0)
                img2 = h_xt.data.cpu().squeeze(0).numpy().transpose(1,2,0)

                flow,mask = opticalflow(img1,img2)

                d1, d2, d3, d4, d5, d6, d7, d8 = net(x_t)
                a1pha1 = PROCESS(d1, x_t)
                del d1, d2, d3, d4, d5, d6, d7, d8

                d1, d2, d3, d4, d5, d6, d7, d8 = net(x_t1)
                a1pha2 = PROCESS(d1, x_t1)
                del d1, d2, d3, d4, d5, d6, d7, d8

                h_xt_features = vgg(h_xt)
                h_xt1_features = vgg(h_xt1)
                x_xt_features = vgg(a1pha1)
                x_xt1_features = vgg(a1pha2)



                # ContentLoss, conv3_2
                content_t = F.mse_loss(x_xt_features[2], h_xt_features[2])
                content_t1 = F.mse_loss(x_xt1_features[2], h_xt1_features[2])
                content_loss = opt.content_weight *  (content_t1 + content_t)
                # StyleLoss
                style_t = 0
                style_t1 = 0
                for ft_y, gm_s in zip(h_xt_features, gram_style):
                    gram_y = gram_matrix(ft_y)
                    style_t += F.mse_loss(gram_y, gm_s.expand_as(gram_y))
                for ft_y, gm_s in zip(h_xt1_features, gram_style):
                    gram_y = gram_matrix(ft_y)
                    style_t1 += F.mse_loss(gram_y, gm_s.expand_as(gram_y))

                style_loss = opt.style_weight * (style_t1 + style_t)

                # # depth loss
                depth_loss1 = F.mse_loss(depth_h_xt, depth_x_t)
                depth_loss2 = F.mse_loss(depth_h_xt1, depth_x_t1)
                depth_loss =  opt.depth_weight * (depth_loss1 + depth_loss2)
                # # TVLoss
                # print(type(s_hxt[layer]),s_hxt[layer].size())
                # tv_loss = TVLoss(h_xt)

                #Long-temprol loss
                if (i-1) % opt.sample_frames == 0:
                    frames0 = h_xt1.cpu()
                    long_img1 = frames0.data.cpu().squeeze(0).numpy().transpose(1,2,0)
                # long_img2 = h_xt.data.cpu().squeeze(0).numpy().transpose(1,2,0)
                long_flow, long_mask = opticalflow(long_img1, img2)

                # Optical flow

                flow = torch.from_numpy(flow).permute(2, 0, 1).unsqueeze(0).to(torch.float32)
                long_flow = torch.from_numpy(long_flow).permute(2, 0, 1).unsqueeze(0).to(torch.float32)

                # print(flow.size())
                # print(h_xt1.size())
                warped = warp(h_xt1.cpu().permute(0,2,3,1), flow ,opt.img_shape[1],opt.img_shape[0]).cuda()
                long_warped = warp(frames0.cpu().permute(0,2,3,1), long_flow ,opt.img_shape[1],opt.img_shape[0]).cuda()
                long_temporal_loss =  F.mse_loss(h_xt , long_mask * long_warped.permute(0,3,1,2))
                # print(warped.size())
                # tv.utils.save_image((warped.permute(0,3,1,2).data.cpu()[0] * 0.225 + 0.45).clamp(min=0, max=1),
                #                     './warped.jpg')
                mask = mask.transpose(2, 0, 1)
                mask = torch.from_numpy(mask).cuda().to(torch.float32)
                # print(mask.shape)
                temporal_loss =  F.mse_loss(h_xt , mask * warped.permute(0,3,1,2))

                temporal_loss = opt.temporal_weight * temporal_loss
                long_temporal_loss = opt.long_temporal_weight * long_temporal_loss

                # Spatial Loss
                spatial_loss = content_loss + style_loss

                Loss = spatial_loss  +  depth_loss + temporal_loss + long_temporal_loss

                Loss.backward(retain_graph=True)
                optimizer.step()
                content_meter.add(float(content_loss.data))
                style_meter.add(float(style_loss.data))
                temporal_meter.add(float(temporal_loss.data))
                long_temporal_meter.add(float(long_temporal_loss.data))
                depth_meter.add(float(depth_loss.data))
                # tv_meter.add(float(tv_loss.data))

                vis.plot('temporal_loss', temporal_meter.value()[0])
                vis.plot('long_temporal_loss', long_temporal_meter.value()[0])
                vis.plot('content_loss', content_meter.value()[0])
                vis.plot('style_loss', style_meter.value()[0])
                vis.plot('depth_loss', depth_meter.value()[0])
                # vis.plot('tv_loss', tv_meter.value()[0])

                if i % 10 == 0:
                    vis.img('input(t)', (x_t.data.cpu()[0] * 0.225 + 0.45).clamp(min=0, max=1))
                    vis.img('output(t)', (h_xt.data.cpu()[0] * 0.225 + 0.45).clamp(min=0, max=1))
                    vis.img('output(t-1)', (h_xt1.data.cpu()[0] * 0.225 + 0.45).clamp(min=0, max=1))
                    print('epoch{},content loss:{},style loss:{},temporal loss:{},long temporal loss:{},depth loss:{},total loss{}'
                          .format(count,content_loss, style_loss,temporal_loss,long_temporal_loss,depth_loss,Loss))
                    # print('epoch{},content loss:{},style loss:{},depth loss:{},total loss{}'
                    #       .format(count,content_loss, style_loss,depth_loss,Loss))


            vis.save([opt.env])
            torch.save(transformer.state_dict(), opt.model_path)

def stylize(**kwargs):
    opt = Config()

    for k_,v_ in kwargs.items():
        setattr(opt, k_, v_)

    style_model = TransformerNet().cuda()
    style_model.load_state_dict(t.load(opt.model_path,))

    transform = transforms.Compose([transforms.ToTensor(),
                                    transforms.Normalize((0.485, 0.456, 0.406),
                                                         (0.229, 0.224, 0.225))])
    video = cv2.VideoCapture(opt.video_path)
    frames = list()
    # 从文件读取视频内容
    # 视频每秒传输帧数
    fps = video.get(cv2.CAP_PROP_FPS)
    # 视频图像的宽度
    frame_width = int(640)
    # 视频图像的长度
    frame_height = int(360)
    # 视频帧数
    frame_count = int(video.get(cv2.CAP_PROP_FRAME_COUNT))

    fourcc = cv2.VideoWriter_fourcc('m','p','4','v')
    out = cv2.VideoWriter('./ablation_4_16_.mp4',fourcc,fps,(frame_width,frame_height))
    n= 0
    while video.isOpened():
        ret, frame = video.read()
        if ret == False:
            break
        n+=1
        frame = cv2.resize(frame, (640, 360))
        # print(ret,frame.shape)
        cv2.imwrite('./ablation/ori/temp%d.jpg'%(n), frame)
        content_image = tv.datasets.folder.default_loader('./ablation/ori/temp%d.jpg'%(n))
        content_image = transform(content_image)
        content_image = content_image.unsqueeze(0).cuda()

        output = style_model(content_image)
        # output = utils.normalize_batch(output)
        tv.utils.save_image((output.data.cpu()[0] * 0.225 + 0.45).clamp(min=0, max=1),
                            './ablation/4/temp%d.jpg'%(n))
        image = cv2.imread('./ablation/4/temp%d.jpg'%(n))
        out.write(image)
        sys.stdout.write('\r>> Converting image %d/%d' % (n, frame_count))
        sys.stdout.flush()

    video.release()
    cv2.destroyAllWindows()





if __name__ == '__main__':
    train()
    # stylize()

