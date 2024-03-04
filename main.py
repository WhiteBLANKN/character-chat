from transformers import (
    DataCollatorForSeq2Seq,
    Trainer
)
from src import (
    parser,
    get_logger,
    process_dataset_and_tokenizer,
    generate_datasets,
    get_model_for_training
)
import os
import configparser

def save_args(model_args, character_args, training_args):
    
    config =configparser.ConfigParser()
    config.read('config.ini')
    
    for name, _args in {
        "model_args":model_args,
        "character_args": character_args,
        "training_args": training_args
        }.items():
        fields = _args.__class__.__dataclass_fields__
        
        for field_name, field in fields.items():
            field_value = getattr(_args, field_name)
            config.set(name, field_name, str(field_value))
    
    with open('config.ini', 'w', encoding="utf-8") as configfile:
        config.write(configfile)

def main():
    model_args, character_args, training_args = parser()
    logger = get_logger(__name__)
    
    logger.info(f"The character to be trained is {character_args.character}")
    logger.info("***********Starting Generating Datasets***********")

    ###############################
    ###GET DATASET AND TOKENIZER###
    ###############################
    
    raw_dataset_path = generate_datasets(character_args)
    train_dataset, tokenizer = process_dataset_and_tokenizer(model_args, character_args, raw_dataset_path)
    data_collator = DataCollatorForSeq2Seq(
        tokenizer=tokenizer,
        pad_to_multiple_of=8,
        padding="longest"
    )
    logger.info("***********Datasets Is Ready!***********")

    ###############################
    ###########GET MODEL###########
    ###############################
    
    model = get_model_for_training(model_args, tokenizer)
    
    ###############################
    ########START TRAINING#########
    ###############################
    
    model.train()
    trainer = Trainer(
        model=model,
        args=training_args,
        data_collator=data_collator,
        train_dataset=train_dataset
    )
    
    trainer.train()
 
    ###############################
    ########SAVE ALL ASSERT########
    ###############################   
    
    model.save_pretrained(
        os.path.join(training_args.output_dir,f'{character_args.character}'),
        save_embedding_layers = True
    )
    tokenizer.save_pretrained(
        os.path.join(training_args.output_dir,f'{character_args.character}/tokenizer')
    )
    
    save_args(model_args, character_args, training_args)
    
    logger.info("**************TRAINING COMPLETE**************")
    logger.info("****USE `chat_termial.py` TO START CHAT******")


if __name__ == '__main__':
    main()