#traning loop
from CAN.BERT import config
from CAN.BERT.model import EmotionTopicClassifier
from CAN.BERT.evaluate import evaluate
import numpy as np
import torch
from torch.utils.tensorboard import SummaryWriter
import os

def train(train_loader, val_loader, pos_weight, run_dir, summary_file):
    torch.manual_seed(config.RANDOM_SEED) #reproducability
    np.random.seed(config.RANDOM_SEED)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(config.RANDOM_SEED)

    device = 'cuda' if torch.cuda.is_available() else 'cpu' #cuda = GPU, else cpu
    emotion_topic_model = EmotionTopicClassifier().to(device) #attach the device, this is in module.nn's to()
    emotion_topic_model.train() # set the module in training mode, also module.nn

    log_dir = os.path.join(run_dir, "tensorboard") # Tensorboard implementation - puts output logs in the run_dir
    writer = SummaryWriter(log_dir=log_dir) # initializes tensorboard

    optimizer = torch.optim.AdamW(emotion_topic_model.parameters(), lr=config.LEARNING_RATE, weight_decay=config.WEIGHT_DECAY)

    emotion_loss_bce = torch.nn.BCEWithLogitsLoss(pos_weight=pos_weight.to(device)) #instantiate first outside the loop with position weights
    topic_loss_ce = torch.nn.CrossEntropyLoss(ignore_index=config.IGNORE_INDEX)

    min_loss = float('inf') #max validation loss for the patience metric
    patience = 2 #epochs allowed without any improvement ^^
    patience_counter = 0
    
    for epoch in range(config.NUM_EPOCHS):

        sum_loss = 0.0
        print(f"epoch {epoch+1} started...")
        for step, batch in enumerate(train_loader): #data loading
            optimizer.zero_grad() #clear the gradients every batch

            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            emotion_labels = batch['emotion_labels'].to(device)
            topic_labels = batch['topic_label'].to(device)


            logits = emotion_topic_model(input_ids, attention_mask) #forward pass
            emotion_logits = logits['emotion_logits']
            topic_logits = logits['topic_logits']

            emotion_mask = (emotion_labels != -100).any(dim=1)
            emotion_loss = emotion_loss_bce(emotion_logits[emotion_mask], emotion_labels[emotion_mask]) if emotion_mask.any() else 0.0 #task-specific loss
            topic_loss = topic_loss_ce(topic_logits, topic_labels) #task-specific loss

            total_loss = emotion_loss + topic_loss # scalar tensor with one value inside (combine loss)

            global_step = epoch * len(train_loader) + step # computes loss per batch and sends results to tensorboard
            writer.add_scalar("Loss/Train_Batch", total_loss.item(), global_step)

            total_loss.backward() #backpropagation

            torch.nn.utils.clip_grad_norm_(emotion_topic_model.parameters(), max_norm=1.0) # gradient clipping, limits the max norm for gradients to add stability.

            optimizer.step() #gradient updates

            sum_loss += total_loss.item() #extracts the scalar number and adds to sum outside loop
            
            if (step+1) % config.LOG_N_STEPS == 0:
                avg = sum_loss / (step + 1)
                print(
                    f"batch {step+1}/{len(train_loader)} | "
                    f"average loss: {avg:.6f}") #print every 100
        
        # patience validation loss
        validation_loss = validate(emotion_topic_model, val_loader, device)


        training_loss = sum_loss / len(train_loader) # average training loss per epoch

        writer.add_scalar("Loss/Train_Epoch", training_loss, epoch) # adds training/validation loss to tensorboard
        writer.add_scalar("Loss/Validation", validation_loss, epoch)

        loss_metrics = (
            f"epoch {epoch+1}/{config.NUM_EPOCHS} | "
            f"training loss: {training_loss:.6f} | "
            f"validation loss: {validation_loss:.6f}"
        )

        print(loss_metrics)
        summary_file.write(loss_metrics + "\n")
        summary_file.flush()
        
        # output confusion matrix
        metrics = evaluate(emotion_topic_model, val_loader, device, run_dir, epoch+1)
        summary_file.write(
            f"Topic Accuracy: {metrics['topic_accuracy']:.6f} | "
            f"Emotion Accuracy: {metrics['emotion_accuracy']:.6f}\n"
        )
        summary_file.flush()

        if validation_loss < min_loss:
            min_loss = validation_loss
            patience_counter = 0
            torch.save(emotion_topic_model.state_dict(), f"{run_dir}/best.pt") # you can save the models dict which has the values of every tensor / parameter values
            summary_file.write("New best model saved.\n")
        else:
            patience_counter+=1
            summary_file.write(f"No improvement. Patience: {patience_counter}/{patience}\n")


        if patience_counter >= patience: # this is our limit
            summary_file.write("Early stopping triggered.\n")
            break
        
    summary_file.close()
    writer.close() # closes tensorboard

def validate(emotion_topic_model, val_loader, device, pos_weight):
    emotion_topic_model.eval() # evaluation mode now, not training

    sum_loss = 0.0

    emotion_loss_bce = torch.nn.BCEWithLogitsLoss(pos_weight=pos_weight.to(device))
    topic_loss_ce = torch.nn.CrossEntropyLoss(ignore_index=config.IGNORE_INDEX)
    
    with torch.no_grad(): #context manager prevents permanent storage (goes away each loop)
        for batch in val_loader:
            #following code is copy pasted from above
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            emotion_labels = batch['emotion_labels'].to(device)
            topic_labels = batch['topic_label'].to(device)
            logits = emotion_topic_model(input_ids, attention_mask)
            emotion_logits = logits['emotion_logits']
            topic_logits = logits['topic_logits']
            emotion_mask = (emotion_labels != -100).any(dim=1)
            emotion_loss = emotion_loss_bce(emotion_logits[emotion_mask], emotion_labels[emotion_mask]) if emotion_mask.any() else 0.0 #task-specific loss
            topic_loss = topic_loss_ce(topic_logits, topic_labels)

            sum_loss += (emotion_loss+topic_loss).item()
    
    emotion_topic_model.train() #back to training mode

    return sum_loss / len(val_loader) #average validation loss per batch