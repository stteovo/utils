import os, torch, onnx


def init_torch_model(torch_model, checkpoint):
    state_dict = torch.load(checkpoint)
    # Adapt the checkpoint
    # for old_key in list(state_dict.keys()):
    #     new_key = '.'.join(old_key.split('.')[1:])
    #     state_dict[new_key] = state_dict.pop(old_key)
    torch_model.load_state_dict(state_dict)
    torch_model.eval()
    return torch_model


def do_export(onnx_save_path, model, in_c=4, shape=[768, 768], checkpoint=None, cuda=False):
    if checkpoint:
        # Initialize model with checkpoint
        model = init_torch_model(model, checkpoint)
    x = torch.randn(1, in_c, shape[0], shape[1])
    inputs = [x[:, :3]]
    input_names = ['input']
    for i in range(3, in_c):
        inputs.append(x[:, i:i+1])
        name = 'mask' + str(i-3) if i != 3 else 'mask'
        input_names.append(name)

    if cuda:
        model.cuda()
        for input in inputs:
            input = input.cuda()

    # Convert model to ONNX
    with torch.no_grad():
        torch.onnx.export(
            model,
            tuple(inputs),
            onnx_save_path,
            do_constant_folding=False,
            opset_version=17,
            input_names=input_names,
            output_names=['diff',
                          'diff_mask'
                          ]),

    onnx_model = onnx.load(onnx_save_path)
    try:
        onnx.checker.check_model(onnx_model)
    except Exception:
        print("Model incorrect")
    else:
        print("Model correct")

def do_export_batch(onnx_save_path, model, shape=[768, 768], checkpoint=None, cuda=False):
    if checkpoint:
        # Initialize model with checkpoint
        model = init_torch_model(model, checkpoint)
    inputs = torch.randn(2, 5, shape[0], shape[1])

    if cuda:
        inputs = inputs.cuda()

    # Convert model to ONNX
    with torch.no_grad():
        torch.onnx.export(
            model,
            tuple(inputs),
            onnx_save_path,
            do_constant_folding=False,
            opset_version=17,
            input_names=["input"],
            output_names=['diff',
                          'diff_mask'
                          ]),

    onnx_model = onnx.load(onnx_save_path)
    try:
        onnx.checker.check_model(onnx_model)
    except Exception:
        print("Model incorrect")
    else:
        print("Model correct")


if __name__ == '__main__':
    from datetime import datetime
    onnx_dir = './onnx/' + datetime.now().strftime('%Y-%m-%d')
    os.makedirs(onnx_dir, exist_ok=True)

    from models.FFC import FFCResNetGenerator as Generator

    generator = Generator(in_c=4, out_c=4, b_linear_up=False, b_testing=True)
    checkpoint = '/data_ssd/doublechin/checkpoints/pipeline/23/ks_diff_in4_s23_384x768/latest_net_0.pth'
    onnx_save_path = os.path.join(onnx_dir, 'ks_diff_in4_s23.onnx')


    do_export(generator, checkpoint, onnx_save_path)




