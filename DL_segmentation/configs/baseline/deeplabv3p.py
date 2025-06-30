from configs.base.loveda import train, test, data, optimizer, learning_rate

config = dict(
    model=dict(
        type='DeepLabV3Plus',
        params=dict(
            encoder_name='resnet50',
            classes=2,
            encoder_weights='imagenet',
            loss=dict(
                #fcloss=dict(gamma=2)
                #ufcloss=dict()
                ce=dict()
                # bceloss=dict()
            )
        )
    ),
    data=data,
    optimizer=optimizer,
    learning_rate=learning_rate,
    train=train,
    test=test
)
