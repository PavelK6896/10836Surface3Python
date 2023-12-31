import os
from transformers import WhisperProcessor, DistilBertTokenizer
import torch
from dataclasses import dataclass
from typing import Any, Dict, List, Union
from datasets import load_dataset, DatasetDict
import evaluate
from transformers import WhisperTokenizer
from transformers import WhisperForConditionalGeneration
from transformers import WhisperFeatureExtractor
from datasets import Audio
import librosa
from transformers import WhisperFeatureExtractor
from transformers import WhisperTokenizer

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
from datasets import Dataset

arr = []
with open("data/i.txt", "r", encoding='utf-8') as file1:
    for line in file1:
        split = line.strip().split(" == ")
        speech, sr = librosa.load("data/" + str(split[0]) + ".wav", sr=16000)
        arr.append({"audio": speech, "transcription": split[1]})

print(arr)
dataset = Dataset.from_list(arr)
newData = dataset.train_test_split(test_size=0.1)
print(newData)

processor = WhisperProcessor.from_pretrained("openai/whisper-base", language="ru", task="transcribe")
tokenizer = WhisperTokenizer.from_pretrained("openai/whisper-base", language="ru", task="transcribe")

model = WhisperForConditionalGeneration.from_pretrained("openai/whisper-base")
feature_extractor = WhisperFeatureExtractor.from_pretrained("openai/whisper-base")
model.config.forced_decoder_ids = None
model.config.suppress_tokens = []

common_voice = newData # .cast_column("audio", Audio(sampling_rate=16000))
print(common_voice)

def prepare_dataset(batch):
    # load and resample audio data from 48 to 16kHz
    audio = batch["audio"]

    # compute log-Mel input features from input audio array
    batch["input_features"] = feature_extractor(audio, sampling_rate=16000).input_features[0]

    # encode target text to label ids
    batch["labels"] = tokenizer(batch["transcription"]).input_ids
    # batch["labels"] = tokenizer(batch["sentence"],truncation=True, padding='max_length', max_length=3500).input_ids
    return batch

common_voice2 = common_voice.map(prepare_dataset, remove_columns=common_voice.column_names["train"])
print(common_voice2)

@dataclass
class DataCollatorSpeechSeq2SeqWithPadding:
    processor: Any

    def __call__(self, features: List[Dict[str, Union[List[int], torch.Tensor]]]) -> Dict[str, torch.Tensor]:
        # split inputs and labels since they have to be of different lengths and need different padding methods
        # first treat the audio inputs by simply returning torch tensors
        input_features = [{"input_features": feature["input_features"]} for feature in features]
        batch = self.processor.feature_extractor.pad(input_features, return_tensors="pt")

        # get the tokenized label sequences
        label_features = [{"input_ids": feature["labels"]} for feature in features]
        # pad the labels to max length
        labels_batch = self.processor.tokenizer.pad(label_features, return_tensors="pt")

        # replace padding with -100 to ignore loss correctly
        labels = labels_batch["input_ids"].masked_fill(labels_batch.attention_mask.ne(1), -100)

        # if bos token is appended in previous tokenization step,
        # cut bos token here as it's append later anyways
        if (labels[:, 0] == self.processor.tokenizer.bos_token_id).all().cpu().item():
            labels = labels[:, 1:]

        batch["labels"] = labels

        return batch

# подборщик
data_collator = DataCollatorSpeechSeq2SeqWithPadding(processor=processor)
metric = evaluate.load("wer")

def compute_metrics(pred):
    pred_ids = pred.predictions
    label_ids = pred.label_ids

    # replace -100 with the pad_token_id
    label_ids[label_ids == -100] = tokenizer.pad_token_id

    # we do not want to group tokens when computing the metrics
    pred_str = tokenizer.batch_decode(pred_ids, skip_special_tokens=True)
    label_str = tokenizer.batch_decode(label_ids, skip_special_tokens=True)

    wer = 100 * metric.compute(predictions=pred_str, references=label_str)

    return {"wer": wer}




from transformers import Seq2SeqTrainingArguments

training_args = Seq2SeqTrainingArguments(
    output_dir="./whisper-small-hi",  # change to a repo name of your choice
    per_device_train_batch_size=16,
    gradient_accumulation_steps=1,  # increase by 2x for every 2x decrease in batch size
    learning_rate=1e-5,
    warmup_steps=5,
    max_steps=20,
    gradient_checkpointing=True,
    fp16=False,
    evaluation_strategy="steps",
    per_device_eval_batch_size=8,
    predict_with_generate=True,
    generation_max_length=225,
    save_steps=10,
    eval_steps=10,
    logging_steps=5,
    report_to=["tensorboard"],
    load_best_model_at_end=True,
    metric_for_best_model="wer",
    greater_is_better=False,
    push_to_hub=False,
)

# padding =‘max_length’
from transformers import Seq2SeqTrainer

trainer = Seq2SeqTrainer(
    args=training_args,
    model=model,
    train_dataset=common_voice2["train"],
    eval_dataset=common_voice2["test"],
    data_collator=data_collator,
    compute_metrics=compute_metrics,
    tokenizer=processor.feature_extractor,
)



from tokenizers import Tokenizer
from tokenizers.models import BPE

# tokenizer.save("tokenizer.json")
# processor.feature_extractor.save_pretrained("./tokenizer/")
tokenizer.save_pretrained("./model")
tokenizer.save_pretrained("./model/tokenizer/")

tokenizer = Tokenizer(BPE(unk_token="[UNK]"))

print("start")
# trainer.train()
#
#
# print(trainer.evaluate())
# trainer.save_model("./model")
