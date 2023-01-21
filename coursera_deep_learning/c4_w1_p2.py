
    # Z1 = tf.keras.layers.Conv2D(
                # filters     = 8,
                # kernel_size = (4,4),
                # strides     = 1 ,
                # padding     = 'SAME'
            # )( input_img )

    # A1 = tf.keras.layers.ReLU()(Z1)

    # P1 = tf.keras.layers.MaxPool2D(
            # pool_size = (8 , 8),
            # strides   = 8,
            # padding   = 'SAME'
        # )(A1)

    # Z2 = tf.keras.layers.Conv2D(
                # filters     = 16,
                # kernel_size = (2,2),
                # strides     = 1 ,
                # padding     = 'SAME'
            # )(P1)

    # A2 = tf.keras.layers.ReLU()(Z2)

    # P2 = tf.keras.layers.MaxPool2D(
            # pool_size = (4 , 4),
            # strides   = 4,
            # padding   = 'SAME'
        # )(A2)

    # F = tf.keras.layers.Flatten()(P2)

    # outputs = tf.keras.layers.Dense(
            # units = 6,
            # activation='softmax'
        # )(F)

