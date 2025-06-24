import torch.nn as nn
import torch, time, os, cv2

from segmentation_models_pytorch.encoders import timm_sknet_encoders
from segmentation_models_pytorch.decoders.upernet.decoder import UPerNetDecoder
from segmentation_models_pytorch.base.heads import SegmentationHead


class Cascade_Unet(nn.Module):
    def __init__(self, num_classes, enc_type='timm-skresnet18',**kwargs):
        super(Cascade_Unet, self).__init__()

        enc_cfg = timm_sknet_encoders.get(enc_type)
        self.encoder = enc_cfg['encoder'](**enc_cfg['params'])

        c_dec_out = 64
        self.decoder = UPerNetDecoder(
            enc_cfg['params']['out_channels'],
            encoder_depth=5,
            pyramid_channels=256,
            segmentation_channels=c_dec_out
        )

        self.cls_head = SegmentationHead(c_dec_out, num_classes, kernel_size=3, activation=None, upsampling=4)

    def forward(self, x):
        x = self.encoder(x)
        x = self.decoder(x)
        seg_map = self.cls_head(x)
        return seg_map
    
    def export_to_onnx(self, 
                       input_size=(1, 3, 768, 768), 
                       pth_file_path=None, 
                       onnx_file_path="cascade_unet.onnx", 
                       b_test_time=False):
        dummy_input = torch.randn(input_size).cuda()  # 假设使用GPU
        self.init_weights(pth_file_path)
        self.eval()

        # 导出模型到ONNX格式
        torch.onnx.export(
            self,
            dummy_input,
            onnx_file_path,
            export_params=True,
            opset_version=16,
            do_constant_folding=True,
            input_names=['input'],   # 输入名称
            output_names=['output'],  # 输出名称
            dynamic_axes=None,
            # dynamic_axes={'input': {0: 'batch_size', 2:'height', 3:'width'},    # 动态批量大小
            #               'output': {0: 'batch_size', 2:'out_height', 3:"out_width"}}  # 动态批量大小
        )
        print(f"模型已成功导出为 {onnx_file_path}")

        if b_test_time:
            # 测试ONNX模型的耗时
            import onnxruntime as ort

            ort_session = ort.InferenceSession(onnx_file_path)
            start_time = time.perf_counter()
            for _ in range(100):  # 测试100次
                ort_session.run(None, {'input': dummy_input.cpu().numpy()})
            end_time = time.perf_counter()
            print(f"ONNX模型推理时间: {(end_time - start_time) * 1000 / 100}ms")


    def init_weights(self, model_file_path=None):
        if model_file_path is not None and os.path.exists(model_file_path):
            self.load_state_dict(torch.load(model_file_path))
        else:
            for m in self.modules():
                if isinstance(m, nn.Conv2d):
                    nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                    if m.bias is not None:
                        nn.init.constant_(m.bias, 0)
                elif isinstance(m, nn.BatchNorm2d):
                    nn.init.constant_(m.weight, 1)
                    nn.init.constant_(m.bias, 0)
                elif isinstance(m, nn.Linear):
                    nn.init.normal_(m.weight, 0, 0.01)
                    nn.init.constant_(m.bias, 0)
        print("模型参数初始化完成")

if __name__ == '__main__':
    from torchinfo import summary
    import torch
    import time

    part = 1
    if part == 0:
        input_data = torch.randn(4, 3, 768, 768).cuda()  # 假设使用GPU
        model = Cascade_Unet(num_classes=9).cuda()  # 将模型移动到GPU
        # summary(model, input_size=(4, 3, 768, 768))

        start_time = time.perf_counter()
        for i in range(100):
            output = model(input_data)
        end_time = time.perf_counter()

        print(f"模型推理时间: {(end_time - start_time) * 1000 / 30}ms")

    elif part == 1:
        model = Cascade_Unet(num_classes=9).cuda()  # 将模型移动到GPU
        model.export_to_onnx(input_size=(1, 3, 768, 768), 
                             pth_file_path="/data_ssd/ay/checkpoint/HandReshape/Seg/seg_v2/latest_net_0.pth",
                             onnx_file_path="/root/group-trainee/ay/HandReshape/dataProcess/handSegModel.onnx", 
                             b_test_time=True)