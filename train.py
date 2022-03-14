import pickle
import pandas as pd
from model.trainer import ModelTrainer
from params import (BATCH_SIZE, FINE_TUNED, IS_CONTINUOUS_TRAIN,
                    IS_SPLIT_INDEXES, LAMDA, LEARNING_RATE,
                    N_EPOCH, PATH_PRETRAINED_MODEL, PENALTY_VALUE,
                    TRAINING_DATA_PERCENT, USE_DETECTION_CONTEXT, PKL_PATH, MODEL_NAME, IS_TRANSFORMER, ADD_CHAR_LEVEL, IS_BERT)

with open(PKL_PATH+f'{MODEL_NAME}.pkl', 'rb') as file:
    data = pd.DataFrame(data = pickle.load(file))
n_samples = data.shape[0]
def main():
    trainer = ModelTrainer (
        model_name=MODEL_NAME,
        n_samples=n_samples,
        training_data_percent=TRAINING_DATA_PERCENT,
        lr=LEARNING_RATE,
        is_continuous_train=IS_CONTINUOUS_TRAIN,
        path_pretrain_model=PATH_PRETRAINED_MODEL,
        is_split_indexs=IS_SPLIT_INDEXES,
        n_epochs=N_EPOCH,
        lam=LAMDA,
        penalty_value=PENALTY_VALUE,
        use_detection_context=USE_DETECTION_CONTEXT,
        is_transformer=IS_TRANSFORMER, 
        add_char_level=ADD_CHAR_LEVEL,
        is_bert = IS_BERT,
        fine_tuned=FINE_TUNED,
        batch_size=BATCH_SIZE
    )
    trainer.fit()


if __name__ == "__main__":
    main()