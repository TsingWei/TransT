import importlib
import torch
import time

tracker_name = 'transt'
config_name = 'transt50'
tracker_module = importlib.import_module('pytracking.tracker.{}'.format(tracker_name))
param_module = importlib.import_module('pytracking.parameter.{}.{}'.format(tracker_name, config_name))
params = param_module.parameters()

tracker_class = tracker_module.get_tracker_class()
tracker = tracker_class(params)

use_gpu = True
torch.cuda.set_device(0)
model = tracker.params.net
model.initialize()

xs = 256
zs = 128

x = torch.randn(1, 3, xs, xs)
z = torch.randn(1, 3, zs, zs)
# zf = torch.randn(1, 96, 8, 8)
if use_gpu:
    model = model.cuda()
    x = x.cuda()
    z = z.cuda()
zf = model.template(z)

T_w = 10  # warmup
T_t = 100  # test
with torch.no_grad():
    for i in range(T_w):
        oup = model.track(x)
    t_s = time.time()
    for i in range(T_t):
        oup = model.track(x)
    torch.cuda.synchronize()
    t_e = time.time()
    print('speed: %.2f FPS' % (T_t / (t_e - t_s)))


print("Done")
