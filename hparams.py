from text import symbols


class Hyperparameters:
    def __init__(self, **kwargs):
        for key, value in kwargs.items():
            setattr(self, key, value)


def create_hparams(local=True, log_output="print"):

    # ESD
    if local:
        training_files = "/home/xzodia/dev/emo-gan/data/ESD/training.csv"
        mixed_training_files = "/home/xzodia/dev/emo-gan/data/ESD/mixed_training.csv"

        validation_files = "/home/xzodia/dev/emo-gan/data/ESD/validation.csv"
        mixed_validation_files = (
            "/home/xzodia/dev/emo-gan/data/ESD/mixed_validation.csv"
        )

        generator_checkpoint = (
            "/home/xzodia/dev/emo-gan/outputs/pretrain_generator/pretrain_4/"
        )
        discriminator_checkpoint = (
            "/home/xzodia/dev/emo-gan/outputs/pretrain_discriminator_ESD/pretrain_7/"
        )

        audio_file_root = "/home/xzodia/dev/emo-gan/data/ESD/"
        log_filename = "logs.log"

    else:
        training_files = "/mnt/scratch/users/mna2002/emo-gan/data/ESD/training.csv"
        mixed_training_files = (
            "/mnt/scratch/users/mna2002/emo-gan/data/ESD/mixed_training.csv"
        )

        validation_files = "/mnt/scratch/users/mna2002/emo-gan/data/ESD/validation.csv"
        mixed_validation_files = (
            "/mnt/scratch/users/mna2002/emo-gan/data/ESD/mixed_validation.csv"
        )

        generator_checkpoint = (
            "/mnt/scratch/users/mna2002/emo-gan/outputs/pretrain_generator/pretrain_4/"
        )
        discriminator_checkpoint = "/mnt/scratch/users/mna2002/emo-gan/outputs/pretrain_discriminator_ESD/pretrain_7/"

        audio_file_root = "/mnt/scratch/users/mna2002/emo-gan/data/ESD/"
        log_filename = "logs.log"

    ## EMNS
    # if local:
    #     training_files = "/home/xzodia/dev/emo-gan/data/EMNS/training.csv"
    #     mixed_training_files = None
    #     validation_files = "/home/xzodia/dev/emo-gan/data/EMNS/validation.csv"
    #     mixed_validation_files = None
    #     audio_file_root = "/home/xzodia/dev/emo-gan/data/EMNS/raw_wavs/"
    #     log_filename = "logs.log"
    #     generator_checkpoint = (
    #         "/home/xzodia/dev/emo-gan/outputs/pretrain_generator/pretrain_4/"
    #     )
    #     discriminator_checkpoint = (
    #         "/home/xzodia/dev/emo-gan/outputs/pretrain_discriminator_ESD/pretrain_7/"
    #     )
    # else:
    #     training_files = "/mnt/scratch/users/mna2002/emo-gan/data/EMNS/training.csv"
    #     mixed_training_files = None
    #     validation_files = "/mnt/scratch/users/mna2002/emo-gan/data/EMNS/validation.csv"
    #     mixed_validation_files = None
    #     audio_file_root = "/mnt/scratch/users/mna2002/emo-gan/data/EMNS/raw_wavs/"
    #     log_filename = "logs.log"
    #     generator_checkpoint = (
    #         "/home/xzodia/dev/emo-gan/outputs/pretrain_generator/pretrain_4/"
    #     )
    #     discriminator_checkpoint = (
    #         "/home/xzodia/dev/emo-gan/outputs/pretrain_discriminator_ESD/pretrain_7/"
    #     )

    if log_output == "log":
        print_log = "log"
    else:
        print_log = "print"

    hparams = Hyperparameters(
        ################################
        # Experiment Parameters        #
        ################################
        epochs=100,
        checkpoint_interval=10,
        seed=1234,
        dynamic_loss_scaling=True,
        fp16_run=False,
        distributed_run=False,
        dist_backend="nccl",
        dist_url="tcp://localhost:54321",
        cudnn_enabled=True,
        cudnn_benchmark=False,
        generator_checkpoint=generator_checkpoint,
        discriminator_checkpoint=discriminator_checkpoint,
        ################################
        # Data Parameters             #
        ################################
        load_mel_from_disk=False,
        training_files=training_files,
        mixed_training_files=mixed_training_files,
        validation_files=validation_files,
        mixed_validation_files=mixed_validation_files,
        audio_file_root=audio_file_root,
        text_cleaners=["english_cleaners"],
        ################################
        # Audio Parameters             #
        ################################
        max_wav_value=32768.0,
        ### EMNS
        ## 48khz sample
        # sampling_rate=48000,
        # filter_length=2048,
        # hop_length=512,
        # win_length=2048,
        # mel_fmax=12000.0,
        # mel_fmin=0.0,
        ### ESD
        ## 16khz sample
        sampling_rate=16000,
        filter_length=2048,
        hop_length=200,
        win_length=800,
        mel_fmax=8000.0,
        mel_fmin=0.0,
        n_mel_channels=80,
        n_channels=3,
        n_frames_per_step=2,
        noise_std=0.005,
        ################################
        # Model Parameters             #
        ################################
        n_symbols=len(symbols),
        # n_emotions=8,
        n_emotions=5,
        symbols_embedding_dim=512,
        emotions_embedding_dim=128,
        # Encoder parameters
        encoder_kernel_size=5,
        encoder_n_convolutions=3,
        encoder_embedding_dim=512,
        text_encoder_dropout=0.5,
        hidden_activation="tanh",
        # Decoder parameters
        predict_spectrogram=False,
        feed_back_last=True,
        n_frames_per_step_decoder=2,
        decoder_rnn_dim=512,
        prenet_dim=[256, 256],
        max_decoder_steps=1000,
        stop_threshold=0.5,
        # p_attention_dropout=0.1,
        # p_decoder_dropout=0.1,
        # Attention parameters
        attention_rnn_dim=512,
        attention_dim=128,
        # Location Layer parameters
        attention_location_n_filters=32,
        attention_location_kernel_size=17,
        # Mel-post processing network parameters
        postnet_n_convolutions=5,
        postnet_dim=512,
        postnet_kernel_size=5,
        postnet_dropout=0.5,
        # Classifier parameters
        classifier_dim=[80, 256, 256, 256, 128],
        classifier_kernel_size=[3, 3, 3, 3],
        classifier_n_convolutions=3,
        classifier_pool_size=3,
        classifier_l1=128,
        classifier_l2=256,
        classifier_cell_units=128,
        classifier_num_linear=768,
        classifier_p=10,
        classifier_time_step=800,
        # classifier_time_step=2048,
        classifier_F1=64,
        classifier_dropout=1,
        ################################
        # Optimization Hyperparameters #
        ################################
        use_saved_learning_rate=False,
        learning_rate=0.0001,
        weight_decay=1e-6,
        grad_clip_thresh=5.0,
        batch_size=8,
        betas=(0.9, 0.999),
        ################################
        # Logs                         #
        ################################
        print_log=print_log,
        log_filename=log_filename,
    )

    return hparams
