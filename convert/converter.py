import os
import torch
import tensorflow as tf
from tensorflow import keras
from tensorflow.python.keras import backend as K

import onnx


def torch2onnx(torch_model, input_tensors, output_path, input_names,
               output_names, keep_initializers=False,
               verify_after_export=False):
    if isinstance(input_tensors, torch.Tensor):
        input_tensors = (input_tensors,)

    torch_model.eval()
    with torch.no_grad():
        output = torch.onnx._export(
            torch_model, input_tensors, output_path,
            export_params=True, verbose=True,
            input_names=input_names, output_names=output_names,
            keep_initializers_as_inputs=keep_initializers,
            opset_version=11)

    if verify_after_export:
        onnx_model = onnx.load(output_path)
        onnx.checker.check_model(onnx_model)

        if not keep_initializers:
            print('keep_initializers=False -> '
                  'Not performing onnx output verification')
            return

        import caffe2.python.onnx.backend as onnx_caffe2_backend

        prepared_backend = onnx_caffe2_backend.prepare(onnx_model)
        W = {name: tensor for name, tensor in zip(input_names, input_tensors)}
        c2_out = prepared_backend.run(W)
        for i in range(len(output)):
            np.testing.assert_almost_equal(
                output[i].data.cpu().numpy(), c2_out[i], decimal=3)


def onnx2keras(onnx_model_path, input_names, output_dir,
               swap_channel_ordering=True):
    from onnx2keras import onnx_to_keras

    onnx_model = onnx.load(onnx_model_path)
    k_model = onnx_to_keras(onnx_model, input_names,
                            change_ordering=swap_channel_ordering,
                            name_policy='renumerate')
    weights = k_model.get_weights()

    K.set_learning_phase(0)

    with K.get_session() as sess:
        init = tf.global_variables_initializer()
        sess.run(init)
        k_model.set_weights(weights)
        tf.saved_model.simple_save(
            sess,
            output_dir,
            inputs={input.name: tensor for input, tensor in zip(
                onnx_model.graph.input, k_model.inputs)},
            outputs={output.name: tensor for output, tensor in zip(
                onnx_model.graph.output, k_model.outputs)})


def onnx2keras_pb(onnx_model_path, input_names, output_names, output_path,
                  swap_channel_ordering=True):
    from onnx2keras import onnx_to_keras
    from convert.utils import freeze_session

    output_dir, filename = os.path.split(output_path)

    onnx_model = onnx.load(onnx_model_path)
    k_model = onnx_to_keras(onnx_model, input_names,
                            change_ordering=swap_channel_ordering,
                            name_policy='renumerate')
    weights = k_model.get_weights()

    K.set_learning_phase(0)

    with K.get_session() as sess:
        init = tf.global_variables_initializer()
        sess.run(init)
        k_model.set_weights(weights)

        frozen_graph = freeze_session(sess, keep_var_names=None, output_names=output_names)
        tf.train.write_graph(frozen_graph, output_dir, filename, as_text=False)


def onnx2pb(onnx_model_path, output_path):
    from onnx_tf.backend import prepare

    model = onnx.load(onnx_model_path)
    tf_rep = prepare(model)
    tf_rep.export_graph(output_path)


def pb2tflite(pb_path, input_node_names, output_node_names, tflite_path):
    converter = tf.compat.v1.lite.TFLiteConverter.from_frozen_graph(
        pb_path, input_node_names, output_node_names)
    tflite_model = converter.convert()

    # converter.experimental_new_converter = True
    # converter.target_spec.supported_ops = [
    #     tf.lite.OpsSet.TFLITE_BUILTINS,
    #     tf.lite.OpsSet.SELECT_TF_OPS]

    with open(tflite_path, 'wb') as f:
        f.write(tflite_model)


def keras2onnx(keras_model, output_path):
    import keras2onnx as keras_to_onnx

    onnx_model = keras_to_onnx.convert_keras(keras_model, keras_model.name)
    keras_to_onnx.save_model(onnx_model, output_path)


def keras2tflite(keras_model_dir, output_path, add_tf_ops=False, allow_fp16=False):
    converter = tf.lite.TFLiteConverter.from_saved_model(
        keras_model_dir)

    if add_tf_ops:
        converter.experimental_new_converter = True
        converter.target_spec.supported_ops = [
            tf.lite.OpsSet.TFLITE_BUILTINS,
            tf.lite.OpsSet.SELECT_TF_OPS]

    if allow_fp16:
        converter.optimizations = [tf.lite.Optimize.DEFAULT]
        converter.target_spec.supported_types = [tf.float16]
    tflite_model = converter.convert()

    with open(output_path, 'wb') as f:
        f.write(tflite_model)


def hdf52savedmodel(hdf5_path, saved_model_dir):
    model = tf.keras.applications.InceptionV3(
        include_top=True,
        weights="imagenet",
        input_tensor=None,
        pooling=None,
        classes=1000,
    )
    model.save(saved_model_dir, save_format='tf')
