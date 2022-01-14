from pathlib import Path
import argparse
import sys
import logging

from datetime import datetime

import tensorflow as tf
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import ModelCheckpoint, ReduceLROnPlateau, CSVLogger, EarlyStopping,TensorBoard

from trainer.utils import ADAMLearningRateTracker, jacc_coef
from trainer.model import model_arch

# logger
model_logger = logging.getLogger(__name__)
model_logger.setLevel(logging.INFO)

formatter = logging.Formatter('%(levelname)s:%(name)s:%(message)s')

model_logger_file_handler = logging.FileHandler('model.log')
model_logger_file_handler.setFormatter(formatter)

model_logger.addHandler(model_logger_file_handler)



def _parse_arguments(argv):
    """Parses command-line arguments."""
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--train_data_path',
        help='train data path',
        type=str, default="/home/jupyter/cloud_detection/data/train_data.csv")
    parser.add_argument(
        '--val_data_path',
        help='validation data path',
        type=str, default="/home/jupyter/cloud_detection/data/val_data.csv")
    parser.add_argument(
        '--batch_size',
        help='model batch size',
        type=int, default=12)
    parser.add_argument(
        '--epochs',
        help='The number of epochs to train',
        type=int, default=10)
    parser.add_argument(
        '--random_state',
        help='random state',
        type=int, default=42)
    parser.add_argument(
        '--starting_learning_rate',
        help='starting learning rate',
        type=float, default=1e-4)
    parser.add_argument(
        '--end_learning_rate',
        help='end learning rate',
        type=float, default=1e-8)
    parser.add_argument(
        '--input_rows',
        help='input image input_rows',
        type=int, default=192)
    parser.add_argument(
        '--input_cols',
        help='input image input_rows',
        type=int, default=192)
    parser.add_argument(
        '--patience',
        help='patience for early_s_patience.ReduceLROnPlateau',
        type=int, default=15)
    parser.add_argument(
        '--decay_factor',
        help='decay_factor for tensorflow.keras.callbacks.ReduceLROnPlateau',
        type=float, default=0.7)
    parser.add_argument(
        '--experiment_name',
        help='experiment_name',
        type=str, default="cloudnet")
    parser.add_argument(
        '--early_s_patience',
        help='tensorflow.keras.callbacks.EarlyStopping patience',
        type=int, default=20)
    parser.add_argument(
        '--num_of_channels',
        help='num_of_channels',
        type=int, default=16)
    parser.add_argument(
        '--num_of_classes',
        help='num_of_classes',
        type=int, default=4)
    parser.add_argument(
        '--reshape',
        help='reshape image and mask to the sampe shape',
        type=bool, default=True)
    parser.add_argument(
        '--quick_test',
        help='run the model on a smaler sample',
        type=bool, default=False)
    parser.add_argument(
        '--train_resume',
        help='resume train or not',
        type=bool, default=False)
    parser.add_argument(
        '--job-dir',
        help='Directory where to save the given model',
        type=str, default='cloud_detection_models/')
    
    return parser.parse_known_args(argv)

def main():
    
    # Get the arguments
    args = _parse_arguments(sys.argv[1:])[0]

    #BATCH_SIZE = args.batch_size
    # SHUFFLE_BUFFER = 10 * BATCH_SIZE
    # RANDOM_STATE = args.random_state
    # AUTOTUNE = tf.data.experimental.AUTOTUNE
    TRAIN_DATA_PATH = args.train_data_path
    VAL_DATA_PATH = args.val_data_path

    #quick_test = args.quick_test
    current_time = datetime.now().strftime("%Y%m%d-%H%M%S")
    experiment_name = f"{args.experiment_name}_{current_time}"
    
    ROOT_DIR = Path.cwd().resolve()
    MODEL_DIR = ROOT_DIR / "models"
    TRAIN_DIR = MODEL_DIR / "train"
    TEST_DIR = MODEL_DIR / "test"
    EXP_DIR = TRAIN_DIR / experiment_name
    
    ORIGINAL_MODEL_WEIGHT_PATH = (MODEL_DIR / "original_weights") / "Cloud-Net_trained_on_38-Cloud_training_patches.h5" # not implemented

    folders = [MODEL_DIR,TRAIN_DIR,TEST_DIR,EXP_DIR]
    for folder in folders:
        if not folder.exists():
            folder.mkdir(parents = False,exist_ok= True)

    MODEL_WEIGHTS_PATH = ROOT_DIR/"model_weights"
    if not MODEL_WEIGHTS_PATH.exists():
        MODEL_WEIGHTS_PATH.mkdir()

    weights_path = MODEL_WEIGHTS_PATH / "weights.{epoch:02d}-{val_loss:.2f}.hdf5"

    random_state = args.random_state

    # hparams
    # starting_learning_rate = args.starting_learning_rate
    # end_learning_rate = args.end_learning_rate
    # epochs = args.epochs # just a huge number. The actual training should not be limited by this value
    # #val_ratio = 0.2
    # patience = args.patience
    # decay_factor = args.decay_factor
    # experiment_name = args.experiment_name
    # early_s_patience = args.early_s_patience

    # params
    input_rows = args.input_rows
    input_cols = args.input_cols
    # img_shape = (input_rows,input_cols)
    num_of_channels = args.num_of_channels
    num_of_classes = args.num_of_classes
    reshape = args.reshape

    # hparams
    batch_size = args.batch_size
    starting_learning_rate = args.starting_learning_rate
    end_learning_rate = args.end_learning_rate
    max_num_epochs = args.epochs # just a huge number. The actual training should not be limited by this value
    patience = args.patience
    decay_factor = args.decay_factor

    early_s_patience = args.early_s_patience
    train_resume = args.train_resume
    
    # log
    model_logger.info("All parameters have been paresed")
    
    # datasets
    train_dataset = load_dataset(file_paths= TRAIN_DATA_PATH, training = True,reshape= reshape, num_epochs=max_num_epochs)
    val_dataset = load_dataset(file_paths= VAL_DATA_PATH, training = False,reshape= reshape)

    # Model
    strategy = tf.distribute.MirroredStrategy()
    model_logger.info('Number of devices: {}'.format(strategy.num_replicas_in_sync))

    with strategy.scope():
        model = model_arch(input_rows=input_rows,
                                           input_cols=input_cols,
                                           num_of_channels=num_of_channels,
                                           num_of_classes=num_of_classes)
        
        model.compile(optimizer=Adam(learning_rate=starting_learning_rate), loss=jacc_coef, metrics=[jacc_coef])
    # model.summary()

    model_checkpoint = ModelCheckpoint(weights_path, monitor='val_loss', save_best_only=True)
    lr_reducer = ReduceLROnPlateau(factor=decay_factor, cooldown=0, patience=patience, min_lr=end_learning_rate, verbose=1)
    csv_logger = CSVLogger(EXP_DIR / '_log_1.log')
    tensorboard = TensorBoard(log_dir= EXP_DIR / 'logs', histogram_freq=0, write_graph=True,write_images=False, write_steps_per_second=False,
                                   update_freq='epoch',profile_batch=0, embeddings_freq=0, embeddings_metadata=None, **kwargs)

    if train_resume:
        model.load_weights(ORIGINAL_MODEL_WEIGHT_PATH)
         model_logger.info("\nTraining resumed...")
    else:
         model_logger.info("\nTraining started from scratch... ")

    model_logger("Experiment name: ", experiment_name)
    model_logger("Input image size: ", (input_rows, input_cols))
    model_logger("Number of input spectral bands: ", num_of_channels)
    model_logger("Learning rate: ", starting_learning_rate)
    model_logger("# Epochs: ", max_num_epochs)
    model_logger("Batch size: ", batch_size, "\n")
    
    model.fit(train_dataset,validation_data = val_dataset,epochs = max_num_epochs,verbose = 1,
             callbacks=[model_checkpoint, lr_reducer, ADAMLearningRateTracker(end_learning_rate), csv_logger,tensorboard])
    
if __name__ == '__main__':
    main()
