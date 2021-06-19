from model.MIRNet import MIRNet
import  torch
import tensorflow as  tf
import onnx
from onnx_tf.backend import prepare
import os
from utils.training_util import save_checkpoint,MovingAverage, load_checkpoint

checkpoint = load_checkpoint("../checkpoints/mir/", False, 'latest')
state_dict = checkpoint['state_dict']
model = MIRNet()
model.load_state_dict(state_dict)
model.eval()

# Converting model to ONNX
print('===> Converting model to ONNX.')
try:
    for _ in model.modules():
        _.training = False

    sample_input = torch.randn(1, 3, 256, 256)

    input_nodes = ['input']
    output_nodes = ['output']

    torch.onnx.export( model, 
                        args=sample_input,
                        f="model.onnx", 
                        export_params=True, 
                        input_names=input_nodes, 
                        output_names=output_nodes,
                        opset_version=10)
    print('Successfull.')
except:
    print('Fail.')
# Converting model to Tensorflow
print('===> Converting model to Tensorflow.')
# try:
onnx_model = onnx.load("model.onnx")
output = prepare(onnx_model)
try:
    os.mkdir('tf_model')
except:
    pass
output.export_graph("tf_model/")
print('Successfull.')
# except:
#     print('Fail.')

# Exporting the resulting model to TFLite
print('===> Exporting the resulting model to TFLite.')
converter = tf.lite.TFLiteConverter.from_saved_model("tf_model")
try:
    converter.add_postprocessing_op = True
    converter.optimizations = [tf.lite.Optimize.DEFAULT]
    converter.experimental_new_converter = True
    converter.target_spec.supported_ops =[tf.lite.OpsSet.TFLITE_BUILTINS, tf.lite.OpsSet.SELECT_TF_OPS]

    tflite_model = converter.convert()
    open("model.tflite", "wb").write(tflite_model)
    print('Sucessfull.')
except:
    print('Failed.')
