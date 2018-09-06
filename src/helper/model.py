# Imports
import numpy as np
import tensorflow as tf

# Build Model-Graph
def model_fn(x, y_true, cnn_shape, dense_shape, dcnn_shape, dense_t_reshape, learning_rate=1e-3): 
    input_shape = x.get_shape()
    batch_size = tf.shape(x)[0]
    
    input_layer_adj = tf.reshape(x,[batch_size, input_shape[1], input_shape[2], 1])
    print(input_layer_adj.get_shape())
    
    # CNN Block
    prev_layer = input_layer_adj
    for i in range(len(cnn_shape)):
        layer_info = cnn_shape[i]
        conv = tf.layers.conv2d(inputs = prev_layer,
                                 filters = layer_info[0],
                                 kernel_size = layer_info[1],
                                 strides = layer_info[2],
                                 padding = layer_info[3],
                                 activation = layer_info[4])
        print(conv.get_shape())
        prev_layer = conv
    
    flat_nodes = np.prod(prev_layer.get_shape()[1:])
    flat = tf.reshape(prev_layer, [batch_size, flat_nodes])
    print(flat.get_shape())
    
    # Note: Maybe move dropout layer to an earlier stage
    drop = tf.layers.dropout(inputs=flat, 
                             rate = 0.3,
                             training = True)
    print(drop.get_shape())
    prev_layer = drop

    for i in range(len(dense_shape)):
        layer_info = dense_shape[i]
        dense = tf.layers.dense(inputs = prev_layer,
                               units = layer_info[0],
                               activation = layer_info[1])
        print(dense.get_shape())
        prev_layer = dense

    # TODO The reshape size as parameter?
    dense_nodes = prev_layer.get_shape()[1]
    if dense_nodes % dense_t_reshape != 0:
        raise ValueError(
            "Dimension mismatch on dense reshape: Can't fit "+str(prev_layer.get_shape()[1])+
            " values into ? x ? x "+str(dense_t_reshape) + " slots evenly")
    
    dense_reshape = tf.reshape(
        prev_layer, [batch_size, dense_nodes//dense_t_reshape, dense_t_reshape, 1])
    print(dense_reshape.get_shape())
    prev_layer = dense_reshape
    
    for i in range(len(dcnn_shape)):
        layer_info = dcnn_shape[i]
        deconv = tf.layers.conv2d_transpose(inputs = prev_layer,
                                            filters = layer_info[0],
                                            kernel_size = layer_info[1],
                                            strides = layer_info[2],
                                            padding = layer_info[3],
                                            activation = layer_info[4])
        print(deconv.get_shape())
        prev_layer = deconv
 
    output_shape = y_true.get_shape()
    print("Last layer:",prev_layer.get_shape())
    prediction = tf.reshape(prev_layer, [batch_size, output_shape[1], output_shape[2]])
    print("Prediction layer:",prediction.get_shape())
    print("Truth layer:",output_shape)

    # Load true labels and compute the loss
    # loss = tf.losses.mean_squared_error(y_true, prediction)
    loss = tf.losses.mean_squared_error(y_true, prediction)
    
    # Init Optimizer for training
    optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate)
    train_op = optimizer.minimize(loss = loss,
                                  global_step = tf.train.get_global_step())
    
    return prediction, loss, train_op, drop


def build_modelgraph(in_shape, out_shape, cnn_shape, dense_shape, dcnn_shape, dense_t_reshape, learning_rate):
    # Build model and init settings
    tf.reset_default_graph()

    # Init model
    model_graph = tf.Graph()
    with model_graph.as_default():
        x = tf.placeholder(tf.float32, shape=[
                           None, in_shape[0], in_shape[1]], name="x")
        y_true = tf.placeholder(
            tf.float32, shape=[None, out_shape[0], out_shape[1]], name="y_true")
        pred, loss, optimizer, drop = model_fn(x,
                                               y_true,
                                               cnn_shape, 
                                               dense_shape,
                                               dcnn_shape,
                                               dense_t_reshape,
                                               learning_rate = learning_rate)
        drop.training = True

        # Init Variable Initializer and Saver
        init_op = tf.global_variables_initializer()
        saver = tf.train.Saver(max_to_keep = 3)

    print([x for x in model_graph.get_operations() if x.type=="Placeholder"])
    important_layers = (x, y_true, pred, loss, optimizer, drop)

    return model_graph, saver, important_layers, init_op
