import tensorflow as tf
from tensorflow.keras import backend as K
# from tensorflow.keras.utils import get_custom_objects

def swish(x):
    return K.sigmoid(x) * x

def gpu_setting(opts):
    if opts['tf_version'] == 2:
        print("Num GPUs Available: ", len(tf.config.list_physical_devices('GPU')))
        gpus = tf.config.list_physical_devices('GPU')
        if gpus:
            try:
                tf.config.set_visible_devices(gpus[0], 'GPU')
                logical_gpus = tf.config.list_logical_devices('GPU')
                print(len(gpus), "Physical GPUs,", len(logical_gpus), "Logical GPU")
            except RuntimeError as e:
                # Visible devices must be set before GPUs have been initialized
                print(e)
    else:  # i.e. tf_version == 1.14
        config = tf.compat.v1.ConfigProto()
        config.gpu_options.allow_growth = True
        gpu_str = opts['gpu_num']
        config.gpu_options.visible_device_list = gpu_str

        # Set the TensorFlow session using tf.compat.v1.Session
        sess = tf.compat.v1.Session(config=config)
        tf.compat.v1.keras.backend.set_session(sess)

        # Update custom objects manually
        custom_objects = {
            'swish': swish
        }
        # tf.keras.utils.register_keras_serializable(custom_objects)
        

    return
